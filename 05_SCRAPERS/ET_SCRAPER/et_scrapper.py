import json
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, TimeoutException
import time

def scrape_fek_urls(driver, catalogue_value, year_value):
    """
    Scrapes Fek URLs for a given legislation catalogue and year, handling pagination.
    """
    base_url = "https://search.et.gr/el/search-legislation/"
    current_url = f"{base_url}?legislationCatalogues={catalogue_value}&selectYear={year_value}"
    
    print(f"\n--- Ξεκινάει scraping για Κατάλογο: {catalogue_value}, Έτος: {year_value} ---")
    
    driver.get(current_url)
    all_urls_for_combination = []

    try:
        # Wait for the page to load and find the "Αναζήτηση" button
        search_button_xpath = "//button[contains(., 'Αναζήτηση')]"
        search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, search_button_xpath))
        )
        search_button.click()

        page_number = 1
        while True:
            print(f"Επεξεργασία σελίδας {page_number} για {catalogue_value}/{year_value}...")
            
            # Wait for the table to appear or update after navigating to a new page
            table_xpath = "//table[@id='listing-items']" 
            
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.XPATH, table_xpath))
                )
                time.sleep(2) # Give a little extra time for content to fully render

                table = driver.find_element(By.XPATH, table_xpath)

                links = table.find_elements(By.XPATH, ".//a[contains(@class, 'fek-link') and starts-with(@href, 'https://search.et.gr/fek/?fekId=')]")
                
                current_page_urls_count = 0
                for link in links:
                    href = link.get_attribute('href')
                    if href:
                        all_urls_for_combination.append(href)
                        current_page_urls_count += 1
                print(f"Βρέθηκαν {current_page_urls_count} URLs στη σελίδα {page_number}.")
            
            except StaleElementReferenceException:
                print("Ανιχνεύτηκε StaleElementReferenceException, προσπαθώ να εντοπίσω ξανά τον πίνακα.")
                # This 'continue' will effectively retry the current page processing
                # Make sure there is an outer break condition to prevent infinite loops if the element never becomes stable.
                continue 
            except TimeoutException:
                print("Χρονικό όριο υπέρβασης αναμονής για τον πίνακα. Πιθανώς τέλος σελιδοποίησης ή σφάλμα φόρτωσης.")
                break
            except Exception as e:
                print(f"Σφάλμα κατά την εξαγωγή URLs από τον πίνακα για {catalogue_value}/{year_value}: {e}")
                break

            # Check for the "Επόμενη" (Next) button for pagination
            next_button = None
            try:
                next_button_xpath = "//button[@aria-label='Επόμενη' and contains(., 'Επόμενη')]" 

                next_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, next_button_xpath))
                )
                
                if next_button and next_button.is_displayed() and next_button.is_enabled():
                    driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                    next_button.click()
                    page_number += 1
                    time.sleep(3) # Wait for the next page to load
                else:
                    print("Το κουμπί 'Επόμενη' δεν βρέθηκε, δεν είναι εμφανές ή δεν είναι ενεργοποιημένο. Τέλος σελιδοποίησης.")
                    break
            except (NoSuchElementException, TimeoutException):
                print("Το κουμπί 'Επόμενη' δεν βρέθηκε μετά την αναζήτηση ή υπήρξε timeout. Τέλος σελιδοποίησης.")
                break
            except Exception as e:
                print(f"Σφάλμα κατά τον έλεγχο/πάτημα του κουμπιού 'Επόμενη' για {catalogue_value}/{year_value}: {e}")
                break
        
    except Exception as e:
        print(f"Προέκυψε σφάλμα κατά την αρχική φόρτωση ή αναζήτηση για {catalogue_value}/{year_value}: {e}")

    print(f"Ολοκληρώθηκε scraping για Κατάλογο: {catalogue_value}, Έτος: {year_value}. Συνολικά URLs: {len(all_urls_for_combination)}")
    return all_urls_for_combination

# Main execution block
if __name__ == "__main__":
    legislation_catalogues = {
        "Νόμος": 1,
        "Προεδρικό Διάταγμα": 2,
        "Πράξη Νομοθετικού Περιεχομένου": 3,
        "Βασιλικό Διάταγμα": 102,
        "Νομοθετικό Προεδρικό Διάταγμα": 501,
        "Νομοθετικό Διάταγμα": 301,
        "Αναγκαστικός Νόμος": 101
    }
    
    years = range(2005, 2026) # 2005 to 2025 inclusive (range end is exclusive)

    output_data = {}
    driver = webdriver.Chrome() # Initialize driver once for all scraping

    try:
        driver.maximize_window() # Maximize window once

        for category_name, catalogue_id in legislation_catalogues.items():
            output_data[category_name] = {}
            for year in years:
                urls = scrape_fek_urls(driver, catalogue_id, year)
                output_data[category_name][str(year)] = urls
                time.sleep(1) # Small delay between year/category combinations to be polite to the server

    except Exception as e:
        print(f"Κύριο σφάλμα: {e}")
    finally:
        driver.quit() # Ensure driver is closed at the very end

    # Save to JSON file
    output_filename = "fek_urls_by_category_year.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"\nΌλα τα URLs αποθηκεύτηκαν στο αρχείο: {output_filename}")