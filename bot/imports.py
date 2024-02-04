import os
import sys
def add_path():
    cd = os.getcwd()
    module_name = "animefication"
    pwd = os.path.join(cd, module_name)
    sys.path.append(pwd)
    

if __name__ == "__main__":
    add_path()
    print(sys.path)