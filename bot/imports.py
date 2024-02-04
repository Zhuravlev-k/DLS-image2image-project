import os
import sys
def add_path():
    cd = os.getcwd()
    # pwd = cd + "/animefication" # unix version
    pwd = cd + "\\animefication" # win version
    sys.path.append(pwd)
    

if __name__ == "__main__":
    add_path()
    print(sys.path)