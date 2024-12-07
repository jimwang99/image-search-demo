# Design Choices

In this document, I layout the choices I've made. We are so lucky living in a world where lots of talented people open-source their work, and many other companies offer free-trial of cloud-based services.

My principles of choosing the stack in this demo:

1. Well documented and easy to use.
2. Open-source is better.
3. Local is better than cloud-based.
4. Scalability is NOT the first priority.

> [!NOTE]
>
> I'm feeling the need to explain point (3) above. I prefer local solutions in this demo because
>
> - Educational purposes
> - Easy to try-out
>
> However, it's not easy, because many AI models require power GPUs which are not always available.

## AI stack choices

### Image to embedding

- [Timm backbone](https://huggingface.co/collections/timm/timm-backbones-6568c5b32f335c33707407f8)

### Text to embedding

- [OpenAI's vector embedding API](https://platform.openai.com/docs/guides/embeddings/embedding-models)

### Image caption

- [OpenAI's API for creating tags and caption of image](https://cookbook.openai.com/examples/tag_caption_images_with_gpt4v)

### Multimodal embedding

Instead of creating different vector space for text and images, multimodal models can map images and descriptive texts into the same vector space. That make it easier for searching.

- [OpenCLIP](https://github.com/mlfoundations/open_clip): open-source, local
- [Google Cloud's multimodal embedding API](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings)

### End-to-end solutions

There are existing end-to-end solutions for image search. Obviously using them are not the purpose of this demo. I'm listing them here just for reference.

- [ChromaDB's multimodal search](https://docs.trychroma.com/guides/multimodal)

### Conclusion

- Use multimodal solution, which will make the entire stack much simpler and more end-to-end.
- Create abstract API, and use OpenCLIP underneath.

## Software stack choices

### Programming language

**Python**, no brainer.

- Universal support by all the tools and services.
- Perfect choice for fast prototyping.
- Can use C++/Rust/Go to rewrite performance critical components later.

### Vector database

**[Milvus](https://github.com/milvus-io/milvus)**

- Open-source (Linux foundation).
- Good documentation and community support.
- Good performance when scale-out.

[ChromaDB](https://github.com/chroma-core/chroma)

* Open-source.
* Super light-weight and beginner friendly.

### API service framework

Flask vs. **FastAPI** vs. Django

- Django is too heavy for this demo.
- FastAPI supports `async` and built-in input validation.
- Web service is not the most important part.

### Frontend framework

Streamlit vs. **Gradio**

* Both are simple and well supported.
* Gradio is made for AI use-cases.
* Frontend is not the key feature in this demo.