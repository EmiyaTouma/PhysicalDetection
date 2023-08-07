import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 初始化图像
fig, ax = plt.subplots()


# 定义动画函数
def animate(i):
    # 生成数据
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x + i/10.0)

    # 清空图像
    ax.clear()

    # 绘制图像
    ax.plot(x, y)


# 创建动画对象
ani = FuncAnimation(fig, animate, frames=100, interval=50)

# 显示动画
plt.show()