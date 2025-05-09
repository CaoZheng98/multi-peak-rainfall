import numpy as np
import math
import json
import zlib
from scipy import stats


class ChicagoRainGenerator:
    def __init__(self, a_base=4079.423, c=0.722, b=21.575, n=0.887):
        self.a_base = a_base  # 暴雨公式基础参数
        self.c = c  # 重现期修正系数
        self.b = b  # 时间修正项
        self.n = n  # 衰减指数

    def _compute_a(self, P):
        """计算综合雨力参数"""
        return self.a_base * (1 + self.c * np.log10(P))/167

    def generate_rain(self, P, T, r, time_step=10):
        """
        生成芝加哥雨型时间序列
        :param P: 重现期(年)
        :param T: 总历时(分钟)
        :param r: 雨峰位置系数 [0.3,0.7]
        :param time_step: 时间分辨率(分钟)
        :return: (时间戳列表, 降雨强度列表) mm/h
        """
        a = self._compute_a(P)
        t_p = math.floor(T * r)
        before = np.arange(0, t_p, time_step)
        after = np.arange(before[-1] + time_step, T + 1, time_step)

        # 生成降雨强度序列
        density = []
        for t in before:
            numerator = ((t_p - t) / r * (1 - self.n) + self.b)
            denominator = ((t_p - t) / r + self.b) ** (self.n + 1)
            temp = a * numerator / denominator * 60  # 转换为mm/h
            density.append(round(temp, 2))

        for t in after:
            numerator = ((t - t_p) / (1 - r) * (1 - self.n) + self.b)
            denominator = ((t - t_p) / (1 - r) + self.b) ** (self.n + 1)
            temp = a * numerator / denominator * 60
            density.append(round(temp, 2))

        timestamps = list(before) + list(after)
        return timestamps, density

    def _generate_sample(self, P, T, r):
        """生成单样本数据"""
        timestamps, rainfall = self.generate_rain(P, T, r)
        sample = {
            "metadata": {
                "P": float(P),
                "T": int(T),
                "r": float(r),
                "max_intensity": float(max(rainfall)),
                "checksum": zlib.crc32(str((P, T, r)).encode()) & 0xffffffff
            },
            "rainfall": [[int(t), float(i)] for t, i in zip(timestamps, rainfall)],
            "flow": []  # 需连接SWMM模拟获取实际流量
        }
        return sample

    def build_dataset(self, dataset_type, num_samples):
        """
        构建数据集
        :param dataset_type: train/val/test
        :param num_samples: 样本数量
        """
        samples = []
        for _ in range(num_samples):
            # 参数采样
            if dataset_type == "train":
                P = np.random.choice([1, 2, 3, 5], p=[0.3, 0.3, 0.25, 0.15])
                T = np.random.choice([60, 120, 180, 240], p=[0.2, 0.3, 0.3, 0.2])
                r = np.clip(np.random.beta(2, 2), 0.3, 0.7)
            elif dataset_type == "val":
                P = round(3 + 7 * np.random.beta(2, 1), 1)
                T = int(120 + 240 * np.random.beta(1, 2))
                r = round(np.random.uniform(0.4, 0.6), 2)
            else:  # test
                P = 5 if np.random.rand() < 0.7 else 10
                T = 60 if P == 5 else 360
                r = 0.3 if T == 60 else 0.7

            # 生成样本
            sample = self._generate_sample(P, T, r)
            samples.append(sample)

        return samples


# # ================= 数据集导出 =================
# if __name__ == "__main__":
#     generator = ChicagoRainGenerator()
#
#     # 生成完整数据集
#     train_data = generator.build_dataset("train", 1200)
#     val_data = generator.build_dataset("val", 300)
#     test_data = generator.build_dataset("test", 100)


class MultiPeakRainGenerator(ChicagoRainGenerator):
    def __init__(self, a_base=4079.423, c=0.722, b=21.575, n=0.887):
        super().__init__(a_base, c, b, n)

    def generate_multi_peak_rain(self, P, T, peaks_config, time_step=10):
        """
        生成多峰降雨时间序列
        :param peaks_config: 峰配置列表，每个元素为(r_pos, intensity_ratio)
                            r_pos: 峰位置系数[0.1,0.9]
                            intensity_ratio: 相对于主峰的强度比例[0.2,1.0]
        """
        base_a = self._compute_a(P)
        timestamps = np.arange(0, T + 1, time_step)
        rainfall = np.zeros_like(timestamps, dtype=float)

        # 生成各子峰
        for i, (r, ratio) in enumerate(peaks_config):
            # 子峰历时计算（保持与主峰相似的历时比例）
            sub_T = int(T * 0.3 * (1 + 0.5 * np.random.rand()))  # 30%±15%历时
            sub_a = base_a * ratio

            # 随机选择子峰位置区间
            start_min = int(T * max(r - 0.15, 0))
            start_max = int(T * min(r + 0.15, 1)) - sub_T
            if start_max <= start_min:
                start = max(int(T * r - sub_T / 2), 0)
            else:
                start = np.random.randint(start_min, start_max)

            # 生成子峰序列
            sub_timestamps, sub_rain = self.generate_rain(
                P=1,  # 强度已通过ratio控制
                T=sub_T,
                r=0.5,  # 子峰居中
                time_step=time_step
            )

            # 强度归一化处理
            sub_rain = np.array(sub_rain) * (ratio ** 0.7)  # 非线性缩放

            # 对齐时间索引
            start_idx = np.searchsorted(timestamps, start)
            end_idx = start_idx + len(sub_rain)

            # 边界处理
            if end_idx > len(timestamps):
                sub_rain = sub_rain[:len(timestamps) - start_idx]
                end_idx = len(timestamps)

            # 叠加降雨强度
            rainfall[start_idx:end_idx] += sub_rain

        # 主峰生成（最后一个配置项为主峰）
        main_r, main_ratio = peaks_config[-1]
        main_T = int(T * 0.6)  # 主峰历时较长
        main_rain = np.array(self.generate_rain(P, main_T, main_r, time_step)[1])
        main_rain *= (main_ratio ** 0.5)  # 主峰强度增强

        # 插入主峰
        main_start = int(T * main_r - main_T / 2)
        main_start = max(main_start, 0)
        main_start_idx = np.searchsorted(timestamps, main_start)
        main_end_idx = main_start_idx + len(main_rain)
        if main_end_idx > len(rainfall):
            main_rain = main_rain[:len(rainfall) - main_start_idx]
            main_end_idx = len(rainfall)
        rainfall[main_start_idx:main_end_idx] += main_rain

        # 后处理
        rainfall = np.clip(rainfall, 0, None)  # 确保非负
        rainfall = self._apply_smoothing(rainfall)  # 平滑处理

        return timestamps.tolist(), np.round(rainfall, 2).tolist()

    def _apply_smoothing(self, rainfall, window_size=3):
        """应用滑动平均平滑"""
        window = np.ones(window_size) / window_size
        return np.convolve(rainfall, window, mode='same')

    def _generate_multi_peak_sample(self, P, T, peaks_config):
        """生成多峰样本"""
        timestamps, rainfall = self.generate_multi_peak_rain(P, T, peaks_config)
        return {
            "metadata": {
                "P": float(P),
                "T": int(T),
                "peaks": [
                    {"position": r, "intensity_ratio": ratio}
                    for r, ratio in peaks_config
                ],
                "checksum": zlib.crc32(str((P, T, peaks_config)).encode()) & 0xffffffff
            },
            "rainfall": [[int(t), float(i)] for t, i in zip(timestamps, rainfall)],
            "flow": []
        }

    def build_multi_peak_dataset(self, dataset_type, num_samples):
        """构建多峰数据集"""
        samples = []
        for _ in range(num_samples):
            # 参数采样
            P, T, _ = self._sample_basic_params(dataset_type)

            # 生成峰配置
            num_peaks = np.random.choice([2, 3], p=[0.7, 0.3])  # 双峰或三峰
            peaks = []
            for _ in range(num_peaks):
                r = np.random.uniform(0.1, 0.9)
                ratio = np.random.beta(2, 2)  # 偏向中等强度
                peaks.append((round(r, 2), round(ratio, 2)))

            # 确保主峰为最后一个
            peaks.sort(key=lambda x: x[1])  # 按强度排序
            samples.append(self._generate_multi_peak_sample(P, T, peaks))

        return samples


# # ================= 使用案例 =================
# if __name__ == "__main__":
#     # 生成双峰降雨案例
#     dual_peak_config = [
#         (0.2, 0.6),  # 第一个峰在30%位置，60%强度
#         (0.8, 1.0),  # 主峰在70%位置
#         (0.6, 0.7)
#
#     ]
#
#     generator = MultiPeakRainGenerator()
#     dual_peak = generator._generate_multi_peak_sample(
#         P=5,
#         T=180,
#         peaks_config=dual_peak_config
#     )
#
#     # 可视化
#     import matplotlib.pyplot as plt
#
#     plt.figure(figsize=(10, 4))
#     plt.plot([t[0] for t in dual_peak["rainfall"]],
#              [i[1] for i in dual_peak["rainfall"]])
#     plt.title("Dual-peak Rainfall Pattern")
#     plt.xlabel("Time (min)")
#     plt.ylabel("Intensity (mm/h)")
#     plt.show()



import numpy as np
import math
import json
import zlib
from scipy import stats
from typing import List, Tuple


class EnhancedRainGenerator(ChicagoRainGenerator):
    def __init__(self, a_base=4079.423, c=0.722, b=21.575, n=0.887):
        super().__init__(a_base, c, b, n)
        self.multi_peak_gen = MultiPeakRainGenerator(a_base, c, b, n)

    def _sample_basic_params(self, dataset_type: str) -> Tuple:
        """采样基础参数（P, T）"""
        if dataset_type == "train":
            P = np.random.choice([1, 2, 3, 5], p=[0.3, 0.3, 0.25, 0.15])
            T = np.random.choice([60, 120, 180, 240], p=[0.2, 0.3, 0.3, 0.2])
        elif dataset_type == "val":
            P = round(3 + 7 * np.random.beta(2, 1), 1)
            T = int(120 + 240 * np.random.beta(1, 2))
        else:  # test
            P = 5 if np.random.rand() < 0.7 else 10
            T = 60 if P == 5 else 360
        return P, T

    def _generate_single_peak(self, dataset_type: str) -> dict:
        """生成单峰样本"""
        P, T = self._sample_basic_params(dataset_type)
        r = np.clip(np.random.beta(2, 2), 0.3, 0.7) if dataset_type == "train" else 0.5
        return self._generate_sample(P, T, r)

    def _generate_multi_peak(self, dataset_type: str) -> dict:
        """生成多峰样本"""
        P, T = self._sample_basic_params(dataset_type)

        # 生成峰配置（保证至少一个主峰）
        num_peaks = np.random.choice([2, 3], p=[0.7, 0.3])
        peaks = []
        for _ in range(num_peaks):
            r = np.random.uniform(0.1, 0.9)
            ratio = np.random.beta(2, 2)  # 强度比例偏向中等值
            peaks.append((round(r, 2), round(ratio, 2)))

        # 确保最后一个为主峰
        peaks.sort(key=lambda x: x[1])
        main_peak = (np.clip(peaks[-1][0], 0.3, 0.7), 1.0)  # 主峰强度固定为1.0
        return self.multi_peak_gen._generate_multi_peak_sample(P, T, peaks[:-1] + [main_peak])

    def build_dataset(self, dataset_type: str, num_samples: int) -> List[dict]:
        """
        改进后的数据集生成方法
        :param dataset_type: train/val/test
        :param num_samples: 总样本数
        :return: 包含单峰和多峰的混合数据集
        """
        samples = []
        peak_ratios = {
            "train": 0.95,  # 训练集95%多峰
            "val": 0.3,  # 验证集30%多峰
            "test": 0.5  # 测试集50%多峰（包含更多复杂情况）
        }

        for _ in range(num_samples):
            # 动态调整多峰比例
            if np.random.rand() < peak_ratios[dataset_type]:
                sample = self._generate_multi_peak(dataset_type)
            else:
                sample = self._generate_single_peak(dataset_type)

            # 数据清洗
            # sample["rainfall"] = self._clean_rainfall(sample["rainfall"])
            samples.append(sample)

        return samples

    def _clean_rainfall(self, rainfall: List[List]) -> List[List]:
        """数据清洗：移除连续零值段，保留至少10分钟间隔"""
        cleaned = []
        zero_counter = 0

        for t, i in rainfall:
            if i == 0:
                zero_counter += 1
                if zero_counter <= 1:  # 最多允许连续1个零值（10分钟）
                    cleaned.append([t, float(i)])
            else:
                zero_counter = 0
                cleaned.append([t, float(i)])

        # 确保首尾各保留10分钟缓冲
        if cleaned[0][1] == 0:
            cleaned = cleaned[1:]
        if cleaned[-1][1] == 0:
            cleaned = cleaned[:-1]
        return cleaned

    def _validate_dataset(self, dataset: List[dict]):
        """数据集验证"""
        # 检查时间分辨率
        for sample in dataset:
            timestamps = [t for t, _ in sample["rainfall"]]
            diffs = np.diff(timestamps)
            assert all(d == 10 for d in diffs), "时间分辨率不符合10分钟要求"

        # 检查多峰分布
        multi_count = sum(len(s["metadata"].get("peaks", [])) > 1 for s in dataset)
        print(f"多峰样本占比：{multi_count / len(dataset):.1%}")


class AdvancedMultiPeakGenerator(MultiPeakRainGenerator):
    def generate_multi_peak_rain(self, P, T, peaks_config, time_step=10):
        """改进的多峰生成方法"""
        # 增加峰间间隔约束
        peaks_config = self._enforce_peak_spacing(peaks_config, T)
        return super().generate_multi_peak_rain(P, T, peaks_config, time_step)

    def _enforce_peak_spacing(self, peaks_config: List[Tuple], T: int) -> List[Tuple]:
        """确保峰间距不小于30分钟"""
        sorted_peaks = sorted(peaks_config, key=lambda x: x[0])
        adjusted = []
        last_pos = -np.inf

        for r, ratio in sorted_peaks:
            min_gap = 30 / T  # 转换为相对位置
            if r - last_pos < min_gap:
                r = last_pos + min_gap + np.random.uniform(0, 0.1)
                r = min(r, 0.9)  # 不超过90%位置
            adjusted.append((r, ratio))
            last_pos = r

        return adjusted


# ================= 使用示例 =================
if __name__ == "__main__":
    generator = EnhancedRainGenerator()

    # 生成数据集
    train_set = generator.build_dataset("train", 500)
    val_set = generator.build_dataset("val", 200)
    test_set = generator.build_dataset("test", 500)

    # 验证数据集
    generator._validate_dataset(train_set)
    generator._validate_dataset(val_set)
    generator._validate_dataset(test_set)

    # 保存示例样本
    with open("rainfall_samples_enhance_multi.json", "w") as f:
        json.dump({
            "train_sample": train_set,
            "val_sample": val_set,
            "test_sample": test_set
        }, f, indent=2)