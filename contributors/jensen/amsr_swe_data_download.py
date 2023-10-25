from datetime import datetime, timedelta
import os
import subprocess

def generate_links(start_year, end_year):
    '''
    Generate a list of download links for AMSR daily snow data files.

    Args:
        start_year (int): The starting year.
        end_year (int): The ending year (inclusive).

    Returns:
        list: A list of download links for AMSR daily snow data files.
    '''
    base_url = "https://n5eil01u.ecs.nsidc.org/AMSA/AU_DySno.001/"
    date_format = "%Y.%m.%d"
    delta = timedelta(days=1)

    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year + 1, 1, 1)

    links = []
    current_date = start_date

    while current_date < end_date:
        date_str = current_date.strftime(date_format)
        link = base_url + date_str + "/AMSR_U2_L3_DailySnow_B02_" + date_str + ".he5"
        links.append(link)
        current_date += delta

    return links

if __name__ == "__main__":
    start_year = 2019
    end_year = 2022

    links = generate_links(start_year, end_year)
    save_location = "/home/chetana/gridmet_test_run/amsr"
    with open("/home/chetana/gridmet_test_run/amsr/download_links.txt", "w") as txt_file:
      for l in links:
        txt_file.write(" ".join(l) + "\n")

    #if not os.path.exists(save_location):
    #    os.makedirs(save_location)

    #for link in links:
    #    filename = link.split("/")[-1]
    #    save_path = os.path.join(save_location, filename)
    #    curl_cmd = f"curl -b ~/.urs_cookies -c ~/.urs_cookies -L -n -o {save_path} {link}"
    #    subprocess.run(curl_cmd, shell=True, check=True)
        # print(f"Downloaded: {filename}")

