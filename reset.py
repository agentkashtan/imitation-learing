from utils import get_follower


def main():
    follower = get_follower()
    follower.connect()
    follower.disconnect()

if __name__ == "__main__":
    main()