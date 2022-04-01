# --------------------------------------------------------------------------------------------------------------------------------------------------
# 作者：       赵敏琨
# 日期：       2021年6月
# 说明：       利用Matplot库交互式画图
# --------------------------------------------------------------------------------------------------------------------------------------------------
import io
import numpy as np
from matplotlib import pyplot as plt
import cv2

# Reference: https://blog.csdn.net/u011361880/article/details/76649222 python函数内部变量通过函数属性实现全局变量
# Reference: https://blog.csdn.net/tb_youth/article/details/105902488 python 判断类是否存在某个属性或方法
# Reference: https://blog.csdn.net/qq_34859482/article/details/80617391 Python--Matplotlib（基本用法）
# Reference: https://blog.csdn.net/ngy321/article/details/80109088 将matplotlib绘制的fig保存在缓存中
# Reference: https://blog.csdn.net/weixin_36366711/article/details/113026818 matplot 图在画布中的位置_matplotlib画布与坐标轴
# Reference: https://blog.csdn.net/littlle_yan/article/details/79204544 基于python的图像格式转换（将RGB图像转换为灰度图像）
# Reference: https://blog.csdn.net/u012206617/article/details/103401276 Python3调试类_io.BytesIO
# Reference: https://blog.csdn.net/weixin_42769131/article/details/91363168 python 使用cv2、io.BytesIO处理图片二进制数据
# Reference: https://blog.csdn.net/geyalu/article/details/50190121 Python OpenCV 图片反色

# Instructions: (Right-handed mouse) Quick Left clicks and move
# to draw connected polyline. When you want to darw new lines
# which not connect to the line before, right click the
# position that your new line starts at.

def on_press(event):

    global fig, ax

    if event.button == 1:
        ax.scatter(event.xdata, event.ydata, linewidth=2)
        on_press.xdata = event.xdata
        on_press.ydata = event.ydata
        if hasattr(on_lift, 'xdata'):
            ax.plot([on_lift.xdata, on_press.xdata],
                    [on_lift.ydata, on_press.ydata],
                    linewidth=20)
        fig.canvas.draw()

    elif event.button == 3:
        on_press.xdata = event.xdata
        on_press.ydata = event.ydata
        on_lift.xdata = event.xdata
        on_lift.ydata = event.ydata


def on_lift(event):

    global fig, ax

    if event.button == 1:
        ax.scatter(event.xdata, event.ydata, linewidth=2)
        on_lift.xdata = event.xdata
        on_lift.ydata = event.ydata
        if hasattr(on_press, 'xdata'):
            ax.plot([on_press.xdata, on_lift.xdata],
                    [on_press.ydata, on_lift.ydata],
                    linewidth=20)
        fig.canvas.draw()

    elif event.button == 3:
        on_press.xdata = event.xdata
        on_press.ydata = event.ydata
        on_lift.xdata = event.xdata
        on_lift.ydata = event.ydata


def inverse_color(image):

    height, width = image.shape
    img2 = image.copy()

    for i in range(height):
        for j in range(width):
            img2[i, j] = (255-image[i, j])
    return img2


def draw():

    global fig, ax

    plt.ioff()

    img = np.ones((512, 512, 3))
    fig = plt.figure(facecolor=[1, 1, 1])
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_lift)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    plt.axis("off")
    plt.show()

    # Access cache
    buffer_ = io.BytesIO()

    # Save and decode
    fig.savefig(buffer_, format="jpg")
    buffer_.seek(0)
    img_NP = np.frombuffer(buffer_.read(), np.uint8)
    img = cv2.imdecode(img_NP, cv2.IMREAD_COLOR)

    # Release cache
    buffer_.close()

    # resize
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_g = inverse_color(img_g)
    # cv2.imshow('1', img_g)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.ion()

    return img_g

# See PyCharm help at https://www.jetbrains.com/help/pycharm/