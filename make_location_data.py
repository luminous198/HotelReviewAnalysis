import pandas as pd
from geopy.geocoders import Nominatim

LOCATION_CACHE = {}
geolocator = Nominatim(user_agent="local_hotel_analysis")

def get_location_info(lat, lng):
    if (lat,lng) in LOCATION_CACHE:
        return LOCATION_CACHE[(lat,lng)]
    try:
        location = geolocator.reverse((lat, lng), exactly_one=True)
        address = location.raw['address']
        city = address.get('city', '')
        country = address.get('country', '')
        LOCATION_CACHE[(lat, lng)] = city, country
        return city, country
    except:
        LOCATION_CACHE[(lat, lng)] = None, None
        return None, None


if __name__ == "__main__":
    filename = r'D:\projects\datasets\booking-com-reviews2-europe\Hotel_Reviews.csv'
    outfile = r'D:\projects\datasets\booking-com-reviews2-europe\Hotel_Reviews_with_city.csv'
    df = pd.read_csv(filename)
    unique_locations = df[['lat', 'lng']].drop_duplicates()
    unique_locations[['City', 'Country']] = unique_locations.apply(lambda row: pd.Series(get_location_info(row['lat'], row['lng'])), axis=1)
    unique_locations.to_csv(outfile,index=False)



