import pathlib
class runCwd_path():
    def __new__(self):
        cwd = pathlib.Path(__file__).parent.parent.absolute()
        return cwd 

def Cwd_path():
    return runCwd_path()

# 取得目前檔案的位置
# from lib.cwdS import Cwd_path
# mazda =Cwd_path()
# print(mazda)
#cwd=str(mazda)

