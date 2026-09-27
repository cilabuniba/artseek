# Data

Evaluation datasets go in `data/external/<name>/`. They are not needed to run the demo.

### [ArtPedia](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=35)

Put `artpedia.json` in `artpedia/` and the images in `artpedia/images/{id}.jpg`, where `id` is the key of the entry in `artpedia.json`. The images are downloaded from the `img_url` field of each entry.

### [AQUA](https://github.com/noagarcia/ArtVQA/tree/master/AQUA)

Download the dataset from the link into `aqua/`. Move the JSON annotation files to `aqua/annotations`, the images to `aqua/images`, and everything else to `aqua/additional`.

### [PaintingForm](https://huggingface.co/datasets/steven16/Painting-Form)

Download the dataset with the Hugging Face CLI into `painting_form/` and unzip the images.

### [ExplainMe](https://github.com/noagarcia/explain-paintings)

Put the annotations in `explain_me/annotations` and copy the SemArt images (the same as AQUA) into `explain_me/images`.
