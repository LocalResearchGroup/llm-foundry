# Run this file as `python modal_script.py --gpu L4`

from modal import Image, App

app = App("quick-start")

def get_stats():
    print("get_stats called")
    return get_stats

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="L4")
    args = parser.parse_args()

    get_stats = app.function(gpu=args.gpu)(get_stats)
    with app.run():
        get_stats.remote()
