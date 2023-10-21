
reset = "\033[0m"  # 重置颜色和样式
red = "\033[91m"   # 红色文本
green = "\033[92m" # 绿色文本
yellow = "\033[93m" # 黄色文本
blue = "\033[94m"  # 蓝色文本

def redPrint(str):
    print(red + str + reset)

def greeenPrint(str):
    print(green + str + reset)

def yellowPrint(str):
    print(yellow + str + reset)



if __name__ == "__main__":
    redPrint("aaaa")
    yellowPrint("aaaa")
    print("aaaa")