import json
import os
import re
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
import time

def clean_filename(title):
    """
    Cleans the title to be used as a filename:
    Replaces '/' with '-', removes leading/trailing spaces, and invalid characters.
    """
    cleaned_title = title.replace("/", "-").strip()
    # Remove any characters that are not alphanumeric, hyphens, or spaces
    cleaned_title = re.sub(r'[^\w\s-]', '', cleaned_title)
    # Replace multiple spaces with a single space
    cleaned_title = re.sub(r'\s+', ' ', cleaned_title)
    return cleaned_title

def download_pdf(pdf_url, save_path):
    """
    Downloads a PDF file from the given URL to the specified path.
    """
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        with open(save_path, 'wb') as pdf_file:
            for chunk in response.iter_content(chunk_size=8192):
                pdf_file.write(chunk)
        print(f"  PDF κατέβηκε επιτυχώς: {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"  Σφάλμα κατά τη λήψη του PDF από {pdf_url}: {e}")
        return False

def extract_fek_details(driver, fek_url): # Removed failed_urls_list from args
    """
    Navigates to a FEK detail page and extracts specified information.
    Returns (fek_data, pdf_download_url). fek_data will be None if critical info is missing.
    pdf_download_url will be None if not found, even if fek_data is present.
    """
    driver.get(fek_url)
    time.sleep(2) # Give some time for the page to render fully

    fek_data = {
        "url": fek_url,
        "Τίτλος": None,
        "Τεύχος": None,
        "Ημερομηνία ΦΕΚ": None,
        "Κατάλογος Νομοθεσίας": None,
        "Ημερομηνία επανακυκλοφορίας": None,
        "Θεματικές Ενότητες": [],
        "Ετικέτες": [],
        "Οντότητες": {},
        "Χρονολόγιο": "Δεν είναι άμεσα εξαγώγιμο από το HTML (δυναμικό περιεχόμενο)."
    }
    
    pdf_download_url = None

    try:
        # Extract basic info
        info_elements_map = {
            "Τίτλος": "//div[@class='listing-info-row']/span[@class='row-title' and text()='Τίτλος']/following-sibling::span[@class='row-data']",
            "Τεύχος": "//div[@class='listing-info-row']/span[@class='row-title' and text()='Τεύχος']/following-sibling::span[@class='row-data']",
            "Ημερομηνία ΦΕΚ": "//div[@class='listing-info-row']/span[@class='row-title' and text()='Ημερομηνία ΦΕΚ']/following-sibling::span[@class='row-data']",
            "Κατάλογος Νομοθεσίας": "//div[@class='listing-info-row']/span[@class='row-title' and text()='Κατάλογος Νομοθεσίας']/following-sibling::span[@class='row-data']",
            "Ημερομηνία επανακυκλοφορίας": "//div[@class='listing-info-row']/span[@class='row-title' and text()='Ημερομηνία επανακυκλοφορίας']/following-sibling::span[@class='row-data']"
        }

        for key, xpath in info_elements_map.items():
            try:
                element = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, xpath))
                )
                fek_data[key] = element.text.strip()
            except TimeoutException:
                fek_data[key] = None # Set to None if not found
            except StaleElementReferenceException:
                # Retry finding the element if it became stale
                try:
                    element = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, xpath))
                    )
                    fek_data[key] = element.text.strip()
                except:
                    fek_data[key] = None

        # Check if Title was found, as it's critical for naming files
        if not fek_data["Τίτλος"]:
            print(f"  Σημαντική προειδοποίηση: Ο Τίτλος δεν βρέθηκε για το URL: {fek_url}. Η σελίδα μπορεί να μην είναι η αναμενόμενη.")
            return None, None # Critical failure, return None for data and PDF URL

        # Extract Θεματικές Ενότητες (Thematic Categories)
        try:
            thematic_tags = WebDriverWait(driver, 5).until(
                EC.presence_of_all_elements_located((By.XPATH, "//ul[@class='tag-list' and not(ancestor::div[@class='listing-tags'])]/li[@class='tag-item option']"))
            )
            fek_data["Θεματικές Ενότητες"] = [tag.text.strip() for tag in thematic_tags if tag.text.strip()]
        except TimeoutException:
            fek_data["Θεματικές Ενότητες"] = []

        # Extract Ετικέτες (Tags)
        try:
            tags = WebDriverWait(driver, 5).until(
                EC.presence_of_all_elements_located((By.XPATH, "//div[@class='listing-tags']//ul[@class='tag-list']/li[@class='tag-item option']"))
            )
            fek_data["Ετικέτες"] = [tag.text.strip() for tag in tags if tag.text.strip()]
        except TimeoutException:
            fek_data["Ετικέτες"] = []

        # Extract Οντότητες (Entities)
        try:
            entity_sections = WebDriverWait(driver, 5).until(
                EC.presence_of_all_elements_located((By.XPATH, "//div[@class='listing-accordion-element' and @data-target='entities']//li[@class='listing-entities']"))
            )
            for section in entity_sections:
                try:
                    entity_type_element = section.find_element(By.XPATH, ".//h3[@class='entity-type']")
                    entity_type = entity_type_element.text.strip()
                    entity_labels = [label.text.strip() for label in section.find_elements(By.XPATH, ".//span[@class='entities-label']") if label.text.strip()]
                    if entity_type and entity_labels:
                        fek_data["Οντότητες"][entity_type] = entity_labels
                except NoSuchElementException:
                    continue
        except TimeoutException:
            fek_data["Οντότητες"] = {}

        # Find PDF URL - Enhanced Logic
        try:
            # 1. Try the most specific XPath based on the provided HTML (aria-label and contains .pdf)
            pdf_link_element = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, "//a[@aria-label='Λήψη ΦΕΚ' and contains(@href, '.pdf')]"))
            )
            pdf_download_url = pdf_link_element.get_attribute('href')
            print(f"  Βρέθηκε σύνδεσμος PDF (aria-label 'Λήψη ΦΕΚ'): {pdf_download_url}")
        except TimeoutException:
            # 2. If not found, try by text content 'Λήψη ΦΕΚ' and contains .pdf
            print(f"  Δεν βρέθηκε σύνδεσμος PDF με aria-label. Αναζήτηση με κείμενο 'Λήψη ΦΕΚ' για το URL: {fek_url}")
            try:
                pdf_link_element = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//a[contains(., 'Λήψη ΦΕΚ') and contains(@href, '.pdf')]"))
                )
                pdf_download_url = pdf_link_element.get_attribute('href')
                print(f"  Βρέθηκε σύνδεσμος PDF (κείμενο 'Λήψη ΦΕΚ'): {pdf_download_url}")
            except TimeoutException:
                # 3. If still not found, try the general search for any PDF link
                print(f"  Δεν βρέθηκε σύνδεσμος PDF με κείμενο 'Λήψη ΦΕΚ'. Αναζήτηση για γενικότερο σύνδεσμο για το URL: {fek_url}")
                try:
                    all_pdf_links = WebDriverWait(driver, 5).until(
                        EC.presence_of_all_elements_located((By.XPATH, "//a[contains(@href, '.pdf')]"))
                    )
                    for link in all_pdf_links:
                        href = link.get_attribute('href')
                        if href and href.endswith('.pdf'):
                            pdf_download_url = href
                            print(f"  Βρέθηκε γενικότερος σύνδεσμος PDF: {pdf_download_url}")
                            break # Found one, no need to search further
                    if not pdf_download_url:
                        print(f"  Προειδοποίηση: Ούτε γενικότερος σύνδεσμος PDF βρέθηκε για το URL: {fek_url}")
                except TimeoutException:
                    print(f"  Προειδοποίηση: Ούτε γενικότερος σύνδεσμος PDF βρέθηκε (timeout) για το URL: {fek_url}")
                except Exception as e:
                    print(f"  Σφάλμα κατά την αναζήτηση γενικότερου συνδέσμου PDF για {fek_url}: {e}")
            except Exception as e:
                print(f"  Σφάλμα κατά την αναζήτηση συνδέσμου PDF με κείμενο για το URL {fek_url}: {e}")
        except Exception as e:
            print(f"  Σφάλμα κατά την αναζήτηση συνδέσμου PDF με aria-label για το URL {fek_url}: {e}")
        
    except Exception as e:
        print(f"  Σφάλμα κατά την εξαγωγή δεδομένων από το URL {fek_url}: {e}")
        return None, None # Return None for both if a general exception occurs

    return fek_data, pdf_download_url

# Main execution block
if __name__ == "__main__":
    input_json_filename = "fek_urls_by_category_year.json"
    output_base_dir = "FEK_Archives"
    failed_urls_filename = "failed_fek_urls.json" # New file for failed URLs

    if not os.path.exists(input_json_filename):
        print(f"Σφάλμα: Το αρχείο '{input_json_filename}' δεν βρέθηκε. Παρακαλώ βεβαιωθείτε ότι έχετε εκτελέσει το προηγούμενο script.")
        exit()

    with open(input_json_filename, 'r', encoding='utf-8') as f:
        all_fek_urls_data = json.load(f)

    # List to store URLs that failed extraction or PDF download
    failed_to_process_urls = []

    # Initialize WebDriver
    driver = webdriver.Chrome()
    driver.maximize_window()

    try:
        for category_name, years_data in all_fek_urls_data.items():
            # Clean category name for folder creation
            cleaned_category_name = clean_filename(category_name)
            
            for year, fek_urls_list in years_data.items():
                # Create directory for category/year
                year_dir = os.path.join(output_base_dir, cleaned_category_name, str(year))
                os.makedirs(year_dir, exist_ok=True)
                
                print(f"\nΕπεξεργασία {len(fek_urls_list)} ΦΕΚ για Κατηγορία: '{category_name}', Έτος: '{year}'")

                for i, fek_url in enumerate(fek_urls_list):
                    # Use a placeholder title for initial path check if actual title isn't available yet
                    # This is a temporary title to construct a potential file path for checking existence
                    temp_fek_id = fek_url.split('=')[-1] # Get the fekId from the URL
                    temp_fek_title_for_path = f"FEK_{temp_fek_id}" 
                    
                    # Construct potential JSON and PDF paths to check if already processed
                    json_output_path_check = os.path.join(year_dir, f"{clean_filename(temp_fek_title_for_path)}.json")
                    pdf_output_path_check = os.path.join(year_dir, f"{clean_filename(temp_fek_title_for_path)}.pdf")

                    # Check if both JSON and PDF files already exist for this FEK
                    if os.path.exists(json_output_path_check) and os.path.exists(pdf_output_path_check):
                        print(f"  ΦΕΚ {i+1}/{len(fek_urls_list)}: {fek_url} - Έχει ήδη επεξεργαστεί. Παράλειψη.")
                        continue # Skip to the next URL if already processed

                    print(f"  Επεξεργασία ΦΕΚ {i+1}/{len(fek_urls_list)}: {fek_url}")
                    
                    details, pdf_download_url = extract_fek_details(driver, fek_url) # No failed_urls_list arg here
                    
                    if details is None: # Critical failure (e.g., title missing, or general page error)
                        failed_to_process_urls.append({"url": fek_url, "reason": "Critical data extraction failed"})
                        print(f"  Αποτυχία εξαγωγής κρίσιμων δεδομένων για το ΦΕΚ: {fek_url}")
                        continue

                    # Use the extracted Title for naming files
                    # If title extraction failed, it will fall back to a generic name
                    fek_title = details.get("Τίτλος", f"Αγνωστος_Τίτλος_{temp_fek_id}")
                    cleaned_fek_title = clean_filename(fek_title)
                    
                    # Update paths with the actual cleaned title
                    json_output_path = os.path.join(year_dir, f"{cleaned_fek_title}.json")
                    pdf_output_path = os.path.join(year_dir, f"{cleaned_fek_title}.pdf")

                    # Save extracted details to JSON
                    try:
                        with open(json_output_path, 'w', encoding='utf-8') as f:
                            json.dump(details, f, ensure_ascii=False, indent=4)
                        print(f"  Δεδομένα αποθηκεύτηκαν: {json_output_path}")
                    except Exception as e:
                        print(f"  Σφάλμα κατά την αποθήκευση JSON για {fek_url}: {e}")
                        failed_to_process_urls.append({"url": fek_url, "reason": f"Failed to save JSON: {e}"})
                        continue

                    # Download PDF if URL found
                    if pdf_download_url:
                        if not download_pdf(pdf_download_url, pdf_output_path):
                            # PDF download failed (e.g., network error, 404 from blob storage)
                            failed_to_process_urls.append({"url": fek_url, "reason": "PDF download failed"})
                            # download_pdf already prints its error message
                    else:
                        # PDF URL was genuinely not found on the page, but other data was extracted
                        print(f"  Σημείωση: Δεν βρέθηκε URL PDF στην σελίδα για το ΦΕΚ: {fek_url}")
                        failed_to_process_urls.append({"url": fek_url, "reason": "PDF URL not found on page"})
                    
                    time.sleep(1) # Small delay between each FEK to be polite to the server

    except Exception as e:
        print(f"Κύριο σφάλμα κατά τη διαδικασία: {e}")
    finally:
        driver.quit() # Ensure driver is closed

    # Save the list of failed URLs to a new JSON file
    if failed_to_process_urls:
        with open(failed_urls_filename, 'w', encoding='utf-8') as f:
            json.dump(failed_to_process_urls, f, ensure_ascii=False, indent=4)
        print(f"\nΟρισμένα URLs απέτυχαν να επεξεργαστούν και καταγράφηκαν στο: {failed_urls_filename}")
    else:
        print("\nΌλα τα URLs επεξεργάστηκαν επιτυχώς. Δεν υπάρχουν αποτυχημένα URLs.")

    print("\nΟλοκληρώθηκε η εξαγωγή δεδομένων και η λήψη PDF.")
