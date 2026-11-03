"""The shared topic list, and how each dataset's labels map onto it.

A label absent from these maps is dropped on purpose. Fuzzy groupings such as
"society", "human interest", "BLACK VOICES" and "IMPACT" describe an angle more
than a topic, so forcing them into a class would teach the model noise.
"""

TOPICS = {
    "business": "Business & Work",
    "politics": "Politics",
    "sport": "Sport",
    "entertainment": "Entertainment & Arts",
    "science_tech": "Science & Technology",
    "health": "Health",
    "crime_law": "Crime & Law",
    "conflict": "Conflict & War",
    "disaster_weather": "Disasters & Weather",
    "environment": "Environment",
    "lifestyle": "Lifestyle & Travel",
    "education": "Education",
    "religion": "Religion",
}
TOPIC_KEYS = list(TOPICS)
TOPIC_TO_ID = {key: i for i, key in enumerate(TOPIC_KEYS)}

LONG_ARTICLE_ONLY_TOPICS = ("conflict", "disaster_weather")

BBC_MAP = {
    "business": "business",
    "entertainment": "entertainment",
    "politics": "politics",
    "sport": "sport",
    "tech": "science_tech",
}

MNDS_MAP = {
    "economy, business and finance": "business",
    "labour": "business",
    "politics": "politics",
    "sport": "sport",
    "arts, culture, entertainment and media": "entertainment",
    "science and technology": "science_tech",
    "health": "health",
    "crime, law and justice": "crime_law",
    "conflict, war and peace": "conflict",
    "disaster, accident and emergency incident": "disaster_weather",
    "weather": "disaster_weather",
    "environment": "environment",
    "lifestyle and leisure": "lifestyle",
    "education": "education",
    "religion and belief": "religion",
}

HUFFPOST_MAP = {
    "BUSINESS": "business",
    "MONEY": "business",
    "POLITICS": "politics",
    "SPORTS": "sport",
    "ENTERTAINMENT": "entertainment",
    "ARTS": "entertainment",
    "ARTS & CULTURE": "entertainment",
    "CULTURE & ARTS": "entertainment",
    "TECH": "science_tech",
    "SCIENCE": "science_tech",
    "WELLNESS": "health",
    "HEALTHY LIVING": "health",
    "CRIME": "crime_law",
    "GREEN": "environment",
    "ENVIRONMENT": "environment",
    "TRAVEL": "lifestyle",
    "FOOD & DRINK": "lifestyle",
    "TASTE": "lifestyle",
    "STYLE": "lifestyle",
    "STYLE & BEAUTY": "lifestyle",
    "HOME & LIVING": "lifestyle",
    "EDUCATION": "education",
    "COLLEGE": "education",
    "RELIGION": "religion",
}
