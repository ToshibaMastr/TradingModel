import numpy as np
import matplotlib.pyplot as plt

def logarithmic_scale(x, mean_val, base_scale=1.0, max_output=4.0):
    ratio = x / mean_val
    sign = np.sign(ratio - 1)
    abs_ratio = np.abs(ratio - 1) + 1e-8
    scaled = sign * np.log1p(abs_ratio * base_scale) * (max_output / np.log(2))
    return np.clip(scaled, -max_output, max_output)

def tanh_scale(x, mean_val, sensitivity=2.0, max_output=4.0):
    normalized = (x - mean_val) / mean_val
    scaled = np.tanh(normalized * sensitivity) * max_output
    return scaled

def sigmoid_scale(x, mean_val, steepness=1.0, max_output=4.0):
    normalized = (x - mean_val) / mean_val
    sigmoid = 2 / (1 + np.exp(-normalized * steepness)) - 1
    return sigmoid * max_output

def arctan_scale(x, mean_val, sensitivity=1.0, max_output=4.0):
    normalized = (x - mean_val) / mean_val
    scaled = np.arctan(normalized * sensitivity) * (2 * max_output / np.pi)
    return scaled

# Пример использования с вашими данными
def demo_scaling_functions():
    # Ваши параметры
    min_val, mean_val, max_val = 307379, 6920058, 109332951

    # Создаем тестовые данные
    test_values = np.array([
        min_val,
        mean_val * 0.5,
        mean_val,
        mean_val * 1.12,
        mean_val * 1.204,
        mean_val * 1.23,
        mean_val * 2,
        mean_val * 10,
        mean_val * 100,
        mean_val * 300,  # в 300 раз больше mean
        mean_val * 500,  # в 500 раз больше mean
        max_val
    ])

    print("Исходные значения и их отношение к mean:")
    for val in test_values:
        ratio = val / mean_val
        print(f"Значение: {val:>12,.0f}, Отношение к mean: {ratio:>8.2f}")

    print("\nРезультаты масштабирования:")
    print("Значение        Отношение   Log_scale   Tanh_scale   Sigmoid_scale   Arctan_scale")
    print("-" * 85)

    for val in test_values:
        ratio = val / mean_val
        log_scaled = logarithmic_scale(val, mean_val, base_scale=0.5)
        tanh_scaled = tanh_scale(val, mean_val, sensitivity=1.5)
        sigmoid_scaled = sigmoid_scale(val, mean_val, steepness=1.2)
        arctan_scaled = arctan_scale(val, mean_val, sensitivity=1.0)

        print(f"{val:>16,.0f}   {ratio:>12.2f}   {log_scaled:>12.2f}   {tanh_scaled:>12.2f}   {sigmoid_scaled:>12.2f}   {arctan_scaled:>12.2f}")

def plot_scaling_functions():
    """Визуализация функций масштабирования"""
    mean_val = 6920058

    # Создаем диапазон значений для визуализации
    x_values = np.logspace(5, 8, 1000)  # от 100k до 100M

    # Применяем все функции
    log_scaled = logarithmic_scale(x_values, mean_val, base_scale=0.5)
    tanh_scaled = tanh_scale(x_values, mean_val, sensitivity=1.5)
    sigmoid_scaled = sigmoid_scale(x_values, mean_val, steepness=1.2)
    arctan_scaled = arctan_scale(x_values, mean_val, sensitivity=1.0)

    # Строим график
    plt.figure(figsize=(12, 8))

    plt.semilogx(x_values, log_scaled, label='Logarithmic Scale', linewidth=2)
    plt.semilogx(x_values, tanh_scaled, label='Tanh Scale', linewidth=2)
    plt.semilogx(x_values, sigmoid_scaled, label='Sigmoid Scale', linewidth=2)
    plt.semilogx(x_values, arctan_scaled, label='Arctan Scale', linewidth=2)

    # Добавляем линию среднего значения
    plt.axvline(x=mean_val, color='red', linestyle='--', alpha=0.7, label='Mean')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.xlabel('Исходные значения')
    plt.ylabel('Масштабированные значения')
    plt.title('Сравнение функций масштабирования')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-4.5, 4.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo_scaling_functions()
    print("\n" + "="*85)
    print("Для визуализации запустите: plot_scaling_functions()")
