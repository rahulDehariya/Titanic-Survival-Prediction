from src.data_loader import load_data

def main():
    df = load_data("data/train.csv")
    print(df.head());

if __name__ == "__main__":
    main()