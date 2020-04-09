states_full = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
               "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
               "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
               "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
               "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
               "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
               "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
               "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]
# if place_type=='city', this list will be used
states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]


def _extract_state(tweet):
    """ Extracts a state code given a tweet object """
    key = ''
    if tweet['place']['place_type'] == 'city':
        state_code = tweet['place']['full_name'].split(",")[
            1].lstrip()
        if state_code in states:
            key = state_code

    elif tweet['place']['place_type'] == 'admin':

        state_name = tweet['place']['full_name'].split(",")[
            0].lstrip()

        if state_name in states_full:
            state_index = states_full.index(state_name)
            state_code = states[state_index]
            key = state_code

    return key


def _tweet_data(tweet, clean_tweet):
    coords = tweet['place']['bounding_box']['coordinates'][0]

    south = coords[0][1]
    north = coords[1][1]
    west = coords[0][0]
    east = coords[2][0]

    location = [south, north, west, east]

    tweet_id = tweet['id']
    raw_tweet = tweet['Text']
    norm_tweet = clean_tweet
    place_name = tweet['place']['full_name']
    place_location = location
    place_type = tweet['place']['place_type']
    state_code = _extract_state(tweet)
    user_id = tweet['user']['id'] if 'id' in tweet['user'] else None
    user_display_name = tweet['user']['name'] if 'name' in tweet['user'] else None
    user_screen_name = tweet['user']['screen_name'] if 'screen_name' in tweet['user'] else None
    user_profile_location = tweet['user']['location'] if 'location' in tweet['user'] else None

    data_record = {'tweet_id': tweet_id, 'raw_tweet': raw_tweet, 'norm_tweet': norm_tweet,
                   'place_name': place_name, 'place_location': place_location, 'place_type': place_type, 'state_code': state_code, 'user_id': user_id, 'user_display_name': user_display_name,
                   'user_screen_name': user_screen_name, 'user_profile_location': user_profile_location}

    return data_record
