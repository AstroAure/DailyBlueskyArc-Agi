from github import Github, Auth
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
from atproto import Client, models, client_utils
import cv2
from selenium import webdriver
import argparse
import io
import os

# Color map for the puzzle grid
colormap = ListedColormap([[0/255, 0/255, 0/255, 1],
                           [30/255, 147/255, 255/255, 1],
                           [249/255, 60/255, 49/255, 1],
                           [79/255, 204/255, 48/255, 1],
                           [255/255, 220/255, 0/255, 1],
                           [153/255, 153/255, 153/255, 1],
                           [229/255, 58/255, 163/255, 1],
                           [255/255, 133/255, 27/255, 1],
                           [135/255, 216/255, 241/255, 1],
                           [146/255, 18/255, 49/255, 1]
                          ])

def get_today_id():
    """Get today's puzzle ID from the ARC-AGI website."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)
    url = "https://arcprize.org/play"
    driver.get(url)
    # Find the puzzle ID in the page
    task_name = driver.find_element("id", "task_name").text
    puzzle_id = task_name.split(':')[1].strip()
    date = driver.find_element("id", "current-date").text.strip()
    driver.quit()
    return puzzle_id, date

def get_puzzle_list(github_token, dataset='evaluation'):
    """Get the list of puzzle IDs from the GitHub repository."""
    g = Github(github_token)  # Log in to GitHub
    repo = g.get_repo('arcprize/ARC-AGI-2')  # Get the repository
    list = repo.get_contents(f"data/{dataset}.txt").decoded_content.decode("utf-8").splitlines()  # Get the names of puzzles
    return list


def display_grid_plt(grid, ax):
    """Display a puzzle grid using matplotlib."""
    ax.pcolor(grid, cmap=colormap, vmin=0, vmax=9, edgecolors='dimgrey', linewidths=0.5, alpha=1)
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    ax.invert_yaxis()

def display_puzzle(puzzle_id, github_token, dataset='evaluation',
                   save=False, show=False, show_answer=False):
    """Display a puzzle with its training and test examples."""
    plt.style.use('dark_background')
    # Get the puzzle content
    g = Github(auth=Auth.Token(github_token))  # Log in to GitHub
    repo = g.get_repo('arcprize/ARC-AGI-2')  # Get the repository
    if type(dataset) is list:
        for ds in dataset:
            try:
                puzzle = json.loads(repo.get_contents(f"data/{ds}/{puzzle_id}.json").decoded_content)
                break
            except:
                continue
    else:
        puzzle = json.loads(repo.get_contents(f"data/{dataset}/{puzzle_id}.json").decoded_content)
    n_train = len(puzzle['train']) # Number of training examples
    # Setting up the figure and grid layout
    pad = 0.1
    cell_size = 3
    n_cols = 4 if show_answer else 3  # Number of columns in the grid
    fig = plt.figure(figsize=(n_cols*(cell_size+pad)-pad, n_train*(cell_size+pad)-pad), constrained_layout=True)
    gs = gridspec.GridSpec(n_train, n_cols, figure=fig, width_ratios=[1]*n_cols, height_ratios=[1]*n_train, hspace=pad, wspace=0.25*pad)
    # Displaying the training examples
    for i in range(n_train):
        ax_input = fig.add_subplot(gs[i, 0])
        display_grid_plt(puzzle['train'][i]['input'], ax_input)
        ax_output = fig.add_subplot(gs[i, 1])
        display_grid_plt(puzzle['train'][i]['output'], ax_output)
        if i == 0:
            ax_input.set_title('Input')
            ax_output.set_title('Output')
        ax_input.text(-0.05, 0.5, f"Example {i+1}", transform=ax_input.transAxes, fontsize='x-large', ha='center', va='center', rotation=90)
    # Displaying the test input
    ax_input = fig.add_subplot(gs[0, 2])
    display_grid_plt(puzzle['test'][0]['input'], ax_input)
    ax_input.set_title('Test - Input')
    # Displaying the test output if wanted
    if show_answer:
        ax_output = fig.add_subplot(gs[0, 3])
        display_grid_plt(puzzle['test'][0]['output'], ax_output)
        ax_output.set_title('Test - Output')
    # Saving and plotting
    if save:
        fig.savefig(f"puzzle_{puzzle_id}.png", dpi=100, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    img_data = io.BytesIO()
    fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
    return img_data.getvalue()


def post_bluesky(bsky_handle, bsky_pwd, puzzle_id, img_data, date=None):
    """Post the puzzle on BlueSky."""
    # Log in to BlueSky
    client = Client()
    client.login(bsky_handle, bsky_pwd)
    # Load image
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), -1)[:,:,[2,1,0,3]]
    h, w, _ = img.shape
    # Prepare the post text
    if date is None:
        post_text = client_utils.TextBuilder().text(f"🤖 Daily ARC-AGI puzzle !\n\nPuzzle: {puzzle_id}")
    else:
        post_text = client_utils.TextBuilder().text(f"🤖 Daily ARC-AGI puzzle !\n\nDate: {date}\nPuzzle: {puzzle_id}\n\nTest your solution ").link("here", "https://arcprize.org/play")
    img_alt = f'Puzzle {puzzle_id} of ARC-AGI 2'
    # Post on BlueSky
    client.send_image(
        text=post_text,
        image=img_data,
        image_alt=img_alt,
        image_aspect_ratio=models.AppBskyEmbedDefs.AspectRatio(height=h, width=w),
    )


def main():
    parser = argparse.ArgumentParser(description="Daily Bluesky ARC-AGI Pipeline")
    parser.add_argument('--github_token', type=str, help='GitHub token for authentication', default=None)
    parser.add_argument('--bsky_handle', type=str, help='Bluesky handle for authentication', default=None)
    parser.add_argument('--bsky_pwd', type=str, help='Bluesky password for authentication', default=None)
    parser.add_argument('--puzzle_id', type=str, default='today', help="Puzzle ID to post, or 'today' or 'random' (default: today)")
    parser.add_argument('--save', help='Save the puzzle image', default=False, action='store_true')
    parser.add_argument('--show', help='Show the puzzle image', default=False, action='store_true')
    parser.add_argument('--show_answer', help='Show the answer in the puzzle image', default=False, action='store_true')
    parser.add_argument('--dataset', type=str, default='evaluation', help='Dataset to use for puzzles (default: evaluation)', choices=['evaluation', 'test'])
    
    args = parser.parse_args()

    if args.github_token is None:
        args.github_token = os.getenv('GITHUB_TOKEN')
    if args.bsky_handle is None:
        args.bsky_handle = os.getenv('BSKY_HANDLE')
    if args.bsky_pwd is None:
        args.bsky_pwd = os.getenv('BSKY_PASSWORD')
    
    print("Starting Daily Bluesky ARC-AGI Pipeline...")
    # Get the puzzle ID
    if args.puzzle_id == 'today':
        dataset = ['evaluation', 'training']
        puzzle_id, date = get_today_id()
    elif args.puzzle_id == 'random':
        dataset = args.dataset
        puzzle_list = get_puzzle_list(args.github_token, dataset=dataset)
        puzzle_id = np.random.choice(puzzle_list, 1, replace=False)[0]
        date = None
    else:
        dataset = args.dataset
        puzzle_id = args.puzzle_id
        date = None
    print(f"Selected puzzle ID: {puzzle_id} from dataset {dataset}")

    # Display the puzzle
    print(f"Displaying puzzle {puzzle_id}...")
    img_data = display_puzzle(puzzle_id=puzzle_id, github_token=args.github_token, dataset=dataset,
                              save=args.save, show=args.show, show_answer=args.show_answer)
    
    # Post the puzzle on BlueSky
    print(f"Posting puzzle {puzzle_id} on BlueSky...")
    post_bluesky(bsky_handle=args.bsky_handle, bsky_pwd=args.bsky_pwd,
                 puzzle_id=puzzle_id, img_data=img_data, date=date)


if __name__ == "__main__":
    main()