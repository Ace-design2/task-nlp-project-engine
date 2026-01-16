import dateparser
from dateparser.search import search_dates


def extract_tasks(text):
	"""
	Split text by 'and' keyword to extract separate tasks.
	The NER model (not zero-shot) will identify entities like TASK, DEADLINE, PRIORITY.
	No Facebook model needed.
	"""
	parts = [p.strip() for p in text.replace(" and ", ",").split(",")]
	tasks = [p for p in parts if p]  # filter empty strings
	return tasks


def parse_deadline(text):
	try:
		matches = search_dates(text)
	except Exception:
		matches = None
	if matches:
		dt = matches[-1][1]
		return dt.isoformat()
	dt = dateparser.parse(text)
	return dt.isoformat() if dt else None


def format_task(t):
	return {
		"task": t,
		"deadline": parse_deadline(t),
		"priority": "medium",
		"category": "general"
	}


def nlp_task_ai(text):
	raw_tasks = extract_tasks(text)
	return [format_task(t) for t in raw_tasks]