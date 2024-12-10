from pathlib import Path

import gradio as gr

from config import config
from backend import BackendServer

from loguru import logger

# Create a backend server
server = BackendServer(config)


# search function for the "Search" button
def search(text_input, image_input):
    logger.debug(f"{text_input=} {image_input=}")
    if image_input is not None:
        # search with image
        p = Path(image_input)
        assert p.is_file(), f"Invalid image file path: {p}"
        ids = server.search_with_image(p, top_k=16)
    elif text_input:
        # search with text
        ids = server.search_with_text(text_input, top_k=16)
    else:
        logger.error("Invalid input: both text and image are empty")
        return []

    results = [server.get_image_uri(id) for id in ids]
    logger.debug(f"Search with image: {results=}")
    return results


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Image Search")
    gr.Markdown("Upload an image or input a text query, then press 'Search'.")

    with gr.Row():
        text_input = gr.Textbox(
            label="Text Query", placeholder="Enter your search query here..."
        )
        image_input = gr.Image(label="Upload Image", type="filepath")

    search_button = gr.Button("Search")
    results_gallery = gr.Gallery(
        label="Search Results",
        columns=[4],
        object_fit="contain",
        height="auto",
        show_label=True,
    )

    # Define the functionality of the search button
    search_button.click(search, [text_input, image_input], results_gallery)

# Launch the app
demo.launch(allowed_paths=[str(Path(__file__).parent.parent / "data")])
