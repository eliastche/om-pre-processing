from pathlib import Path
import pandas as pd
def read_market_price_file(filepath):
    """
    Reads a market price file in tab-separated format and converts it to a nested dictionary.
    
    Expected input format:
    Season | Region | Average PV (ore/kWh) | Period
    Fall   | O_Vestavind1 | 192.3148551 | 2030
    ...
    
    Output format:
    {
        "Vestavind1": {
            "2030": {
                "Fall": 192.3148551,
                "Summer": 172.6992945,
                ...
            },
            ...
        },
        ...
    }
    
    Args:
        filepath (str): Path to the market price file
        
    Returns:
        dict: Nested dictionary with structure: region -> year -> season -> price
    """
    # Read the file as tab-separated
    df = pd.read_csv(filepath)
    
    # Initialize the result dictionary
    market_price = {}
    
    # Process each row
    for _, row in df.iterrows():
        season = row['Season']
        region = row['Region']
        price = row['Average PV (ore/kWh)']
        period = str(row['Period'])
        
        # Extract region name from format "O_Vestavind1" -> "Vestavind1"
        region_name = region.split('_')[1]  # Remove O_ prefix
        
        # Initialize nested structure if needed
        if region_name not in market_price:
            market_price[region_name] = {}
        if period not in market_price[region_name]:
            market_price[region_name][period] = {}
        
        # Store the price
        market_price[region_name][period][season] = price
    
    return market_price


def convert_ore_kwh_to_knok_mwh(price_ore_kwh):
    """
    Convert price from øre/kWh to kNOK/MWh.
    
    Conversion factors:
    - 1 øre = 0.01 NOK
    - 1 kNOK = 1000 NOK
    - 1 MWh = 1000 kWh
    
    So: X øre/kWh * 0.01 NOK/øre * 1000 kWh/MWh / 1000 NOK/kNOK = X * 0.01 kNOK/MWh
    
    Args:
        price_ore_kwh (float): Price in øre/kWh
        
    Returns:
        float: Price in kNOK/MWh
    """
    return price_ore_kwh * 0.01


def get_season_from_time(t):
    """
    Determine the season based on the decimal part of time t.
    
    Season mapping:
    - Spring: 0.00 to 0.24
    - Summer: 0.25 to 0.49
    - Fall:   0.50 to 0.74
    - Winter: 0.75 to 0.99
    
    Args:
        t (float): Time value (e.g., 2.3 for t=2.3)
        
    Returns:
        str: Season name ('Spring', 'Summer', 'Fall', 'Winter')
    """
    decimal_part = t % 1.0
    
    if decimal_part < 0.25:
        return "Spring"
    elif decimal_part < 0.50:
        return "Summer"
    elif decimal_part < 0.75:
        return "Fall"
    else:
        return "Winter"

def get_market_year_from_time(t):
    """
    Determine the market year based on the integer part of time t.
    
    Year mapping (0-year basis):
    - 0-4 years → 2030
    - 5-9 years → 2035
    - 10-14 years → 2040
    - etc.
    
    Args:
        t (float): Time value (e.g., 2.3 for t=2.3)
        
    Returns:
        str: Year string (e.g., '2030', '2035')
    """
    integer_part = int(t)
    year_offset = (integer_part // 5) * 5
    base_year = 2030
    return str(base_year + year_offset)


def get_market_price(t, region_name, market_price_dict):
    """
    Get the market price for a given time and region.
    
    Args:
        t (float): Time value (e.g., 2.3 for t=2.3)
        region_name (str): Region name (e.g., 'Vestavind1', 'Nordavind')
        market_price_dict (dict): Market price dictionary with structure:
                                  region -> year -> season -> price (in øre/kWh)
        
    Returns:
        float: Market price in kNOK/MWh, or None if not found
    """
    season = get_season_from_time(t)
    year = get_market_year_from_time(t)
    
    try:
        price_ore_kwh = market_price_dict[region_name][year][season]
        return convert_ore_kwh_to_knok_mwh(price_ore_kwh)
    except KeyError:
        # Return None if the specific market data is not available
        return None


tech_market_data = read_market_price_file("C:\\Users\\IFE13253\\OneDrive - Institutt for Energiteknikk\\Documents\\OffshoreRisk\\RiskSimulation\\MonteCarlo-PostProcces\\Results\\Shadow power price by region- tech.csv")
inc_market_data = read_market_price_file("C:\\Users\\IFE13253\\OneDrive - Institutt for Energiteknikk\\Documents\\OffshoreRisk\\RiskSimulation\\MonteCarlo-PostProcces\\Results\\Shadow power price by region - inc.csv")

print(tech_market_data)


# print ("Hello World!\n")

# def chiaras_function():
#     # This is a function to count the amount of letters in the string
#     text = input("Please enter a string: ")
#     count = 0
#     for char in text:
#         if char.isalpha():
#             count += 1
#     return count

# print(chiaras_function())

# def even_or_odd():
#     # This function checks if a number is even or odd
#     number = int(input("Please enter an number: "))
#     if number % 2 == 0:
#         return "Even"
#     else:
#         return "Odd"

# print(even_or_odd())

# def asciivalue():
#     # This function returns the ASCII value of a character
#     char = input("Please enter a character: ")
#     return ord(char)

# print(asciivalue())

# def ischar():
#     # This function is given an ASCII value and returns theS character
#     ascii_value = int(input("Please enter an ASCII value: "))
#     return chr(ascii_value)                 
# print (ischar())
