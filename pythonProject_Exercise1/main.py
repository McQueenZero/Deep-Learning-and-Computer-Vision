# --------------------------------------------------------------------------------------------------------------------------------------------------
# 作者：       赵敏琨
# 日期：       2021年5月
# 说明：       实验一：基础篇
# 任务1：    图像的基本处理：读取与保存、裁剪与拼接、旋转与对称、放大与缩小、灰度图转换以及灰度直方图统计、加入不同类型噪声、图片的批量读取与按顺序保存；
# 任务2：    图像中的目标可视化：根据附件中提供的图片和标注框信息，对图片中的主要目标进行标注；
# 任务3：    在任务2的基础上，对图片进行旋转、对称等操作之后，对目标标记框的坐标进行准确变换，并将结果记录到txt文档之中，格式和提供的标签内容格式相同，注明旋转角度、对称方式。
# --------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np  # 导入numpy包
import cv2  # 导入opencv包
import matplotlib.pyplot as plt  # 导入matplot的pyplot包
from skimage import util  # 导入skimage包
import os  # 导入os包
import glob  # 导入glob包

#   ----------任务选择----------
print("任务1：图像基本操作，请输入'BO'")  # Basic Operations
print("任务2：目标可视化，请输入'OV'")  # Object Visualization
while 1:
    TASK = input('请选择任务：')
    if TASK == 'BO' or TASK == 'bo':
        break
    elif TASK == 'OV' or TASK == 'ov':
        break
    else:
        print('非法，请重新输入')

#   ----------任务1----------
if TASK == 'BO' or TASK == 'bo':

    #   图像读取、显示、保存
    # Reference: https://zhuanlan.zhihu.com/p/44959772

    imSrc = cv2.imread("./Input Images/lena.png", 1)
    cv2.imshow("Original Image", imSrc)  # show the Image
    print('Press OTHER Key to continue')
    print("Press 's' to save image")
    k = cv2.waitKey(0) & 0xFF
    if k == 27:  # wait for key ESC to exit
        cv2.destroyWindow("Original Image")  # destroy specific window
    elif k == ord('s'):  # wait for key 'S' to save and exit
        cv2.imwrite("./Output Images/lena.jpg", imSrc)  # write image into a file
        cv2.destroyWindow("Original Image")

    #   图像裁剪、拼接
    # ATTENTION: NOT SAME AS MATLAB, PYTHON or C style programming language
    # array's index starts from 0, instead of 1

    # get rows, columns, channels of source image
    rows, cols, chs = imSrc.shape
    # up-left part of the image
    imPart1 = imSrc[1 - 1:int(rows / 2), 1 - 1:int(cols / 2)]
    cv2.imshow("Part of Image", imPart1)
    print('Press a Key to continue')
    cv2.waitKey(0)
    cv2.destroyWindow("Part of Image")
    # down-right part of the image
    imPart2 = imSrc[int(rows / 2):rows, int(cols / 2):cols]
    # Join two parts horizontally
    imPartsJoin = np.hstack([imPart1, imPart2])
    cv2.imshow("Join of Images", imPartsJoin)
    print('Press a Key to continue')
    cv2.waitKey(0)
    cv2.destroyWindow("Join of Images")

    #   图像旋转、对称
    # Reference: https://zhuanlan.zhihu.com/p/47273624
    # Reference: https://blog.csdn.net/pnnngchg/article/details/79420357 python numpy数组中冒号的使用
    # Reference: https://blog.csdn.net/csxiaoshui/article/details/65446125 旋转变换

    # FUNCTION cv2.warpAffine:
    # warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst
    #     .   @brief Applies an affine transformation to an image.
    #     .   The function warpAffine transforms the source image
    #     using the specified matrix M.

    # FUNCTION cv2.getRotationMatrix2D:
    # getRotationMatrix2D(center, angle, scale) -> retval
    #     .   @brief Calculates an affine matrix of 2D rotation.
    #     .   The function calculates the rotation matrix, which you can
    #     specify the rotation center, rotation angle, image scale factor.

    M = cv2.getRotationMatrix2D((int(rows / 2), int(cols / 2)), 45, 0.707)
    imRotate = cv2.warpAffine(imSrc, M, (rows, cols))
    cv2.imshow("Rotation of Image", imRotate)
    print('Press a Key to continue')
    cv2.waitKey(0)
    cv2.destroyWindow("Rotation of Image")

    # Matlab's reverse:  end:-1:1 , Python's reverse:  cols:0:-1
    # Problem of Python's numpy style reverse(Also a difference from Matlab):
    # It contains the start index but doesn't contain the end index.
    # Actually, the col won't include the index of 0
    # I have to use 'hstack' to fix it
    imSym = imSrc[0:rows, cols:0:-1]    # 512x511x3
    imSym = np.hstack([imSym, imSym[:, 0:1]])    # 512x512x3
    # print(imSym.shape)
    cv2.imshow("Symmetry of Image", imSym)
    print('Press a Key to continue')
    cv2.waitKey(0)
    cv2.destroyWindow("Symmetry of Image")

    #   图像放大、缩小
    # Reference: https://blog.csdn.net/weixin_43730228/article/details/84979285

    # FUNCTION cv2.resize:
    # resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst
    #     .   @brief Resizes an image.
    #     .
    #     .   The function resize resizes the image src down to or up to the specified size. Note that the
    #     .   initial dst type or size are not taken into account.

    imResizeUP = cv2.resize(imSrc, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    imResizeDown = cv2.resize(imSrc, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    cv2.imshow("Image Zoom In", imResizeUP)
    print('Press a Key to continue')
    cv2.waitKey(0)
    cv2.destroyWindow("Image Zoom In")
    cv2.imshow("Image Zoom Out", imResizeDown)
    print('Press a Key to continue')
    cv2.waitKey(0)
    cv2.destroyWindow("Image Zoom Out")

    #   灰度图转换、灰度直方图统计
    # Reference: https://blog.csdn.net/weixin_39881922/article/details/80889344

    # FUNCTION cv2.cvtColor:
    # cvtColor(src, code[, dst[, dstCn]]) -> dst
    #     .   @brief Converts an image from one color space to another.
    #     .
    #     .   The function converts an input image from one color space to another. In case of a transformation
    #     .   to-from RGB color space, the order of the channels should be specified explicitly (RGB or BGR). Note
    #     .   that the default color format in OpenCV is often referred to as RGB but it is actually BGR (the
    #     .   bytes are reversed). So the first byte in a standard (24-bit) color image will be an 8-bit Blue
    #     .   component, the second byte will be Green, and the third byte will be Red. The fourth, fifth, and
    #     .   sixth bytes would then be the second pixel (Blue, then Green, then Red), and so on.
    imGray = cv2.cvtColor(imSrc, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Image", imGray)
    print('Press a Key to continue')
    cv2.waitKey(0)
    cv2.destroyWindow("Gray Image")

    # Reference: OpenCV3-Computer-Vision-with-Python-Cookbook, P.141

    hist, bins = np.histogram(imGray, 256, [0, 255])
    plt.figure(num='Histogram of Gray Image')
    plt.fill(hist)
    plt.xlabel('pixel value')
    print('Close the figure window to continue')
    plt.show()

    #   加入不同类型噪声
    # Reference: https://blog.csdn.net/chicken3wings/article/details/100985820

    imNoiseGAS = util.random_noise(imSrc, mode='gaussian')
    imNoiseSP = util.random_noise(imSrc, mode='s&p')
    cv2.imshow("Image with Gaussian Noise", imNoiseGAS)
    print('Press a Key to continue')
    cv2.waitKey(0)
    cv2.destroyWindow("Image with Gaussian Noise")
    cv2.imshow("Image with Salt&Pepper Noise", imNoiseSP)
    print('Press a Key to continue')
    cv2.waitKey(0)
    cv2.destroyWindow("Image with Salt&Pepper Noise")

    #   图片批量读取、按序保存
    # Reference: https://blog.csdn.net/qq_40755643/article/details/84330562

    # Convert input images to the size of 512x512
    # then write into output folder as same name

    def convert_jpg(jpgfile, outdir, width=512, height=512):
        src = cv2.imread(jpgfile, cv2.IMREAD_ANYCOLOR)
        try:
            dst = cv2.resize(src, (width, height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(outdir, os.path.basename(jpgfile)), dst)
        except Exception as e:
            print(e)


    while 1:
        WRITEFLAG = input('请输入批量缩放结果图片是否保存(ON/OFF)：')
        if WRITEFLAG == 'ON' or WRITEFLAG == 'on':
            break
        elif WRITEFLAG == 'OFF' or WRITEFLAG == 'off':
            break
        else:
            print('非法，请重新输入')

    if WRITEFLAG == 'ON' or WRITEFLAG == 'on':
        for jpgfile in glob.glob("./Input Images/*.jpg"):
            convert_jpg(jpgfile, "./Output Images")
    else:
        print('批量保存已关闭')

#   ----------任务2----------
if TASK == 'OV' or TASK == 'ov':

    #   根据dataset数据对图片目标进行标注（图像和文档的批量操作）
    # Reference: https://blog.csdn.net/weixin_43593330/article/details/89882187 numpy.loadtxt() 用法
    # Reference: https://blog.csdn.net/xjp_xujiping/article/details/81604882 Python glob()函数的作用和用法
    # Reference: https://blog.csdn.net/smxjant/article/details/93614544
    #            http://c.biancheng.net/view/2196.html 字符串、列表、元组
    # Reference: https://blog.csdn.net/weixin_39881922/article/details/80076369 目标检测中怎么将画好框的图片保存下来
    # Reference: https://blog.csdn.net/qq_44864262/article/details/106889508 Python matplotlib画坐标点并且以文本内容标记
    # Reference: https://blog.csdn.net/qq_40323256/article/details/112148766 Python中的数组（列表）
    # Reference: https://blog.csdn.net/alansss/article/details/84978672 python下的opencv画矩形和文字注释
    # Reference: https://zhuanlan.zhihu.com/p/47273624 OpenCV, 图像的几何变换
    # Reference: https://blog.csdn.net/a373595475/article/details/79580734 NumPy - 线性代数, dot 两个数组的点积; 数组的连接
    # Reference: https://blog.csdn.net/qq_37828488/article/details/100024924 写入txt文本
    # Reference: https://blog.csdn.net/u011520181/article/details/83933325 Python 用 OpenCV 画点和圆 (2)
    # Reference: https://blog.csdn.net/baidu_29244931/article/details/80278832 python 元组与列表相互转换
    # Reference: https://blog.csdn.net/qq_41895190/article/details/82905657 OpenCV-Python图片叠加与融合，cv2.add与cv2.addWeighted的区别
    # Reference: https://blog.csdn.net/elecjack/article/details/50920318 将numpy array由浮点型转换为整型
    # Reference: https://blog.csdn.net/BF02jgtRS00XKtCx/article/details/114464489 Python 中删除文件的几种方法

    while 1:
        WRITEFLAG = input('请输入坐标变换结果是否批量保存(ON/OFF)：')
        if WRITEFLAG == 'ON' or WRITEFLAG == 'on':
            if os.path.exists("./dataset/labels/counterclockwise 45 deg/"):
                for txtfile in glob.glob("./dataset/labels/counterclockwise 45 deg/*.txt"):
                    os.remove(txtfile)
                print('TXT files deleted successfully')
                # delete txt files generated before
            if os.path.exists("./dataset/labels/mirror symmetry/"):
                for txtfile in glob.glob("./dataset/labels/mirror symmetry/*.txt"):
                    os.remove(txtfile)
                print('TXT files deleted successfully')
                # delete txt files generated before
            break
        elif WRITEFLAG == 'OFF' or WRITEFLAG == 'off':
            break
        else:
            print('非法，请重新输入')

    for Text_Index in glob.glob("./dataset/labels/*.txt"):
        Image_dir = "./dataset/images/" + Text_Index[17] + ".jpg"   # Image path
        Text_CCW45d_dir = "./dataset/labels/counterclockwise 45 deg/" + Text_Index[17] + ".txt"
        Text_MS_dir = "./dataset/labels/mirror symmetry/" + Text_Index[17] + ".txt"
        Image_Obj = cv2.imread(Image_dir)                           # Read Image
        # get rows, columns, channels of source image
        rows, cols, chs = Image_Obj.shape
        # get rotation matrix whose size is 2x3
        M = cv2.getRotationMatrix2D((int(rows / 2), int(cols / 2)), 45, 0.707)
        point_Ro_map = np.zeros((rows, cols, chs), np.uint8)   # Zero grayscale image, used for drawing rotated points
        point_Sym_map = np.zeros((rows, cols, chs), np.uint8)   # Zero grayscale image, used for drawing symmetrical points

        with open(Text_Index, 'r') as fr:  # Open Label File
            label_data = np.loadtxt(fr.name)  # Read Label File
            # print(label_data)
            for Category_Index in range(np.size(label_data, 0)):
                Category_n = int(label_data[Category_Index, 0])  # Category Number
                if Category_n == 0:
                    Category = 'people'
                elif Category_n == 1:
                    Category = 'ball'
                elif Category_n == 2:
                    Category = 'dog'
                elif Category_n == 3:
                    Category = 'car'
                else:
                    Category = 'unknown'
                # print("类别是：", Category)

                Crd_xArray = []  # Initialize Edge x-Coordinate as NONE
                Crd_yArray = []  # Initialize Edge y-Coordinate as NONE
                Crd_Ro_Array = [Category_n]     # Initialize Rotated Coordinate as its Category Number
                Crd_Sym_Array = [Category_n]    # Initialize Symmetrical Coordinate as its Category Number
                point_Ro_list = []  # Initialize Rotated Edge Point list as NONE
                point_Sym_list = []  # Initialize Symmetrical Edge Point list as NONE
                for Coordinate_Index in range(1, 8, 2):
                    Coordinate = [label_data[Category_Index, Coordinate_Index],
                                  label_data[Category_Index, Coordinate_Index+1]]  # 坐标
                    # plt.scatter(Coordinate[0], Coordinate[1])   # plot an Edge Point
                    # plt.annotate(str(Coordinate), xy=Coordinate)   # annotate an Edge Point
                    # print(Coordinate)     # a single Edge Point
                    Crd_xArray.append(Coordinate[0])
                    Crd_yArray.append(Coordinate[1])

                    Coordinate.append(1)    # Augmented Coordinate
                    Coordinate_NP = np.array(Coordinate)    # (NumPy Type) Augmented Coordinate
                    # print(Coordinate_NP)
                    # (NumPy Type) Rotated Coordinate
                    Coordinate_NP_Rotated = np.round(np.dot(M, Coordinate_NP))
                    Coordinate_NP_Rotated = Coordinate_NP_Rotated.astype(int)   # Trans to INT type
                    Coordinate_Rotated = Coordinate_NP_Rotated.tolist()
                    Coordinate_NP_Symmetry = np.round([rows - Coordinate_NP[0], Coordinate_NP[1]])
                    Coordinate_NP_Symmetry = Coordinate_NP_Symmetry.astype(int)
                    Coordinate_Symmetry = Coordinate_NP_Symmetry.tolist()
                    # plt.scatter(Coordinate_Rotated[0], Coordinate_Rotated[1])   # plot an Edge Point
                    # plt.annotate(str(Coordinate_Rotated), xy=Coordinate_Rotated)   # annotate an Edge Point
                    # plt.scatter(Coordinate_Symmetry[0], Coordinate_Symmetry[1])   # plot an Edge Point
                    # plt.annotate(str(Coordinate_Symmetry), xy=Coordinate_Symmetry)   # annotate an Edge Point
                    Crd_Ro_Array.append(Coordinate_Rotated)
                    point_Ro_list.append(tuple(Coordinate_Rotated))    # tuple list
                    Crd_Sym_Array.append(Coordinate_Symmetry)
                    point_Sym_list.append(tuple(Coordinate_Symmetry))  # tuple list

                print("旋转后的标签数据是：", Crd_Ro_Array)
                print("镜面对称后的标签数据是：", Crd_Sym_Array)
                if WRITEFLAG == 'ON' or WRITEFLAG == 'on':
                    with open(Text_CCW45d_dir, "a") as fw:
                        fw.write(str(Crd_Ro_Array[0]))
                        fw.write(' ')
                        # Crd_Ro_Array's 1~4 elements ↓
                        for item in Crd_Ro_Array[1:5]:
                            fw.write(str(item[0]))
                            fw.write(' ')
                            fw.write(str(item[1]))
                            fw.write(' ')
                        fw.write('\r')

                    with open(Text_MS_dir, 'a') as fw:
                        fw.write(str(Crd_Sym_Array[0]))
                        fw.write(' ')
                        # Crd_Sym_Array's 1~4 elements ↓
                        for item in Crd_Sym_Array[1:5]:
                            fw.write(str(item[0]))
                            fw.write(' ')
                            fw.write(str(item[1]))
                            fw.write(' ')
                        fw.write('\r')
                else:
                    print('批量保存已关闭')

                # plt.show()  # Visualize all 4 Edge Points
                # print("旋转后的边缘坐标是：", point_list)  # check rotated coordinates
                for point in point_Ro_list:
                    cv2.circle(point_Ro_map, point, 1, (255, 255, 255), 4)
                for point in point_Sym_list:
                    cv2.circle(point_Sym_map, point, 1, (255, 255, 255), 4)

                Coordinate_UL = (int(min(Crd_xArray)), int(min(Crd_yArray)))
                Coordinate_DR = (int(max(Crd_xArray)), int(max(Crd_yArray)))
                # Get Up-Left & Down-Right Point Coordinates
                Image_Obj_Box = cv2.rectangle(Image_Obj, Coordinate_UL, Coordinate_DR, (0, 255, 0), 2)  # Paint box
                cv2.putText(Image_Obj_Box, Category, Coordinate_UL, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Write Label

        # plt.show()  # Visualize all Edge Points

        Image_Rotated = cv2.warpAffine(Image_Obj_Box, M, (rows, cols))
        Image_Rotated = cv2.addWeighted(Image_Rotated, 1, point_Ro_map, 1, 0)

        Image_Symmetrical = Image_Obj_Box[0:rows, cols:0:-1]
        Image_Symmetrical = np.hstack([Image_Symmetrical, Image_Symmetrical[:, 0:1]])  # 512x512x3
        Image_Symmetrical = cv2.addWeighted(Image_Symmetrical, 1, point_Sym_map, 1, 0)

        cv2.imshow("Objective_Image with box", Image_Obj_Box)
        cv2.imshow("Objective_Image_Rotated with box", Image_Rotated)
        cv2.imshow("Objective_Image_Symmetrical with box", Image_Symmetrical)
        print('Press a Key to continue')
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
