from qdrant_client import QdrantClient, models
from datasets import load_from_disk
from tqdm import tqdm
from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
import torch
from PIL import Image
import os
import lovely_tensors as lt

lt.monkey_patch()


class ColQwen2Qdrant:
    def __init__(
        self, pretrained_model_name_or_path: str | os.PathLike, collection_name: str
    ):
        """Create a ColQwen2 model connected to the Qdrant collection

        Args:
            pretrained_model_name_or_path (str | os.PathLike): Pretrained model name or path
            collection_name (str): Qdrant collection name
        """
        # Model instantiation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ColQwen2.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.processor = ColQwen2Processor.from_pretrained(
            pretrained_model_name_or_path,
            max_num_visual_tokens=768,
        )
        self.model = self.model.eval()
        print("Model instantiated")

        # Client instantiation
        self.collection_name = collection_name
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost")
        self.client = QdrantClient(url=qdrant_url, prefer_grpc=True, timeout=180)
        info = self.client.get_collection(self.collection_name)
        print(f"Client started with the following collection: {info}")

    def embed(self, texts: list[str] = None, images: list[Image.Image] = None):
        """Embed texts or images

        Args:
            texts (list[str], optional): List of text queries to embed. Defaults to None.
            images (list[Image.Image], optional): List of image queries to embed. Defaults to None.

        Returns:
            list: List of embeddings
        """
        assert (
            texts is not None or images is not None
        ), "Either texts or images must be provided"
        if texts is not None and images is not None:
            assert len(texts) == len(
                images
            ), "Texts and images must have the same length"
        text_inputs = (
            self.processor.process_queries(texts).to(self.device)
            if texts is not None
            else None
        )
        image_inputs = (
            self.processor.process_images(images).to(self.device)
            if images is not None
            else None
        )

        with torch.no_grad():
            if text_inputs is None:
                return self.model(**image_inputs).cpu().float().numpy().tolist()
            if image_inputs is None:
                return self.model(**text_inputs).cpu().float().numpy().tolist()
            inputs = image_inputs
            inputs["input_ids"] = torch.cat(
                (
                    inputs["input_ids"][:, :-6],
                    text_inputs["input_ids"],
                    inputs["input_ids"][:, -2:],
                ),
                dim=1,
            )
            inputs["attention_mask"] = torch.cat(
                (
                    inputs["attention_mask"][:, :-6],
                    text_inputs["attention_mask"],
                    inputs["attention_mask"][:, -2:],
                ),
                dim=1,
            )
            # inputs["input_ids"] = torch.cat(
            #     (
            #         text_inputs["input_ids"][:, :2],
            #         inputs["input_ids"][:, 3:-6],
            #         text_inputs["input_ids"][:, 2:],
            #     ), dim=1
            # )
            # inputs["attention_mask"] = torch.cat(
            #     (
            #         text_inputs["attention_mask"][:, :2],
            #         inputs["attention_mask"][:, 3:-6],
            #         text_inputs["attention_mask"][:, 2:],
            #     ), dim=1
            # )
            outputs = self.model(**inputs)
            embeddings = (
                outputs[:, -(text_inputs["input_ids"].shape[1] + 2) : -2]
                .cpu()
                .float()
                .numpy()
                .tolist()
            )
            # embeddings = (
            #     torch.cat((outputs[:, :2], outputs[:, -text_inputs["input_ids"].shape[1] + 2:]), dim=1)
            #     .cpu()
            #     .float()
            #     .numpy()
            #     .tolist()
            # )
            return embeddings

    def query(
        self,
        embeddings: list,
        prefetch_limit: int = 500,
        limit: int = 10,
        prefetch_oversampling: float = 2.0,
        oversampling: float = 2.0,
    ):
        """Query the dataset

        Args:
            embeddings (list): List of embeddings to query
        """
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=models.Prefetch(
                query=embeddings,
                using="pooled",
                limit=prefetch_limit,
                params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=False,
                        oversampling=prefetch_oversampling,
                    )
                ),
            ),
            query=embeddings,
            using="full",
            limit=limit,
            with_payload=True,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=oversampling,
                )
            ),
            timeout=120,
        )

        return response
