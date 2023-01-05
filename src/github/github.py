import time
import os

def push_to_git()->None:
    """
    The goal of this function is to push automatically
    the files to github
    
    Arguments:
        None 
        
    Returns:
        None
    """

    os.system("git status")
    time.sleep(5)
    os.system("git add --a")
    time.sleep(5)
    os.system("git commit -m 'automatic_github_push' --no-verify")
    time.sleep(8)
    os.system("git push")
    time.sleep(10)

if __name__=="__main__":
    push_to_git()