import os

from dotenv import load_dotenv

load_dotenv()


def main():
    print("Hello")
    print(os.environ.get("OPENAI_API_KEY")[:10])


if __name__ == "__main__":
    main()
