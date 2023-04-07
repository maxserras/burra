"""Contains the queries used to filter the data in Argilla for diverse purposes"""

AVOID_TRANSLATION_FLAGGED = """NOT _exists_:metadata.tr-flag-1-instruction AND NOT _exists_:metadata.tr-flag-2-input 
AND NOT _exists_:metadata.tr-flag-2-output"""
AVOID_TRANSLATION_ERRORS = "NOT annotated_by:same-instruction-auto"
TAGGED_ITEMS = "status:Validated"


def merge_queries(*queries):
	"""Merges the given queries into a single query"""
	# Add parenthesis to each query
	queries = [f"({query})" for query in queries]
	return " AND ".join(queries)
