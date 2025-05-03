from langchain_milvus import Milvus
import logging
from typing import (
    List,
    Optional,
)
from langchain_core.documents import Document
from pymilvus import Collection
from pymilvus.milvus_client.milvus_client import MilvusClient

logger = logging.getLogger(__name__)

# Milvus 필요한 함수 Custom
class CustomMilvus(Milvus):
    def search_by_metadata(
            self, expr: str, fields: Optional[List[str]] = None, limit: int = 0
    ) -> List[Document]:
        """
        Searches the Milvus vector store based on metadata conditions.

        This function performs a metadata-based query using an expression
        that filters stored documents without vector similarity.

        Args:
            expr (str): A filtering expression (e.g., `"city == 'Seoul'"`).
            fields (Optional[List[str]]): List of fields to retrieve.
                                          If None, retrieves all available fields.

        Returns:
            List[Document]: List of documents matching the metadata filter.
        """
        from pymilvus import MilvusException

        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        # Default to retrieving all fields if none are provided
        if fields is None:
            fields = self.fields

        try:
            if limit > 0:
                results = self.col.query(expr=expr, output_fields=fields, limit=limit)
            else:
                # limit 설정 없는 경우, no limit
                results = self.col.query(expr=expr, output_fields=fields)
            return [
                Document(page_content=result[self._text_field], metadata={k: v for k, v in result.items() if k != 'vector'})
                for result in results
            ]
        except MilvusException as e:
            logger.error(f"Metadata search failed: {e}")
            return []

    def get_collection_name(self):
        """
        get collection name

        Returns:
            str: Name of the collection.

        """
        if isinstance(self.col, Collection):
            return self.col.name
        return None

    def drop_collection(self):
        """
        Drops this collection.
        """
        if isinstance(self.col, Collection):
            self.col.drop()
            self.col = None

    def get_collections(self):
        """
        get collection names in this db client

        Returns:
            List[String]: List of the collection names
        """
        if isinstance(self.client, MilvusClient):
            return self.client.list_collections()
        return None
