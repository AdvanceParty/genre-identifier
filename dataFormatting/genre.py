genres = {
    'Action': ['action'],
    'Adult': ['adult'],
    'Adventure': ['adventure'],
    'Animation': ['animation'],
    'Biography': ['biography'],
    'Comedy': ['comedy'],
    'Crime': ['crime'],
    'Doc': ['doc', 'documentary'],
    'Drama': ['drama'],
    'Family': ['family'],
    'Fantasy': ['fantasy'],
    'Noir': ['noir', 'film-noir', 'filmnoir'],
    'History': ['history'],
    'Horror': ['horror'],
    'Musical': ['musical'],
    'Mystery': ['mystery'],
    'Romance': ['romance'],
    'SciFi': ['scifi', 'sci-fi'],
    'Thriller': ['thriller'],
    'War': ['war'],
    'Western': ['western'],
    'Short': ['short'],
    'News': ['news'],
    'Sport': ['sport'],
    'Talkshow': ['talk-show', 'talkshow'],
    'Music': ['music']
}


def get_names():
    return genres.keys()


def normalise(term):
    lc = term.lower()
    match = False

    for key, values in genres.items():
        if lc in values:
            match = key
            break

    # if not match:
        # print(f'No genre found for {term} -- {lc}')

    return match if match else 'unknown'
