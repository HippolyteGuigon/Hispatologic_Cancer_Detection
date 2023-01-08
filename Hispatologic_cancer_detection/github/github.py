import time
import os


def push_to_git() -> None:
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

def continuous_push_to_git(seconds_to_wait=300, number_iteration=300)->None:
    """
    The goal of this function is to automatize the 
    push to git to have a push every seconds_to_wait
    
    Arguments:
        -seconds_to_wait: float: The seconds to wait 
        before launching a new push to github 
        -number_iteration: int: The number of times
        to repeat the push to git
        
    Returns:
        None
    """

    counter_push=0

    while counter_push<number_iteration:
        time.sleep(seconds_to_wait)
        push_to_git()
        counter_push+=1



if __name__ == "__main__":
    continuous_push_to_git()
