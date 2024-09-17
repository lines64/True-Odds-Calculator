import pytesseract
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import re
import pandas as pd

# Path to Tesseract executable (make sure to change this to the correct path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to preprocess image for better OCR results
def preprocess_image(image_path):
    image = Image.open(image_path)
    grayscale_image = ImageOps.grayscale(image)
    enhancer = ImageEnhance.Contrast(grayscale_image)
    enhanced_image = enhancer.enhance(2.0)
    return enhanced_image

# Function to extract text from an image using Tesseract OCR
def extract_text_from_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    config = "--psm 6"
    extracted_text = pytesseract.image_to_string(preprocessed_image, config=config)
    return extracted_text

def parse_text(extracted_text):
    pattern = r"([A-Za-z\s]+)\s+(\d+\.\d{2})"
    matches = re.findall(pattern, extracted_text)

    teams_odds = []
    for match in matches:
        team_name = match[0].strip()
        odd = float(match[1])
        teams_odds.append((team_name, odd))

    teams_odds.sort(key=lambda x: x[1])
    return teams_odds

def trueodds(o, method):
    m = np.sum(1.0 / o, axis=1) - 1
    n = o.shape[1]

    if method == 'EM':
        to = o * (m[:, None] + 1)
        p = 1.0 / to
        c = None

    elif method == 'MPTO':
        to = (n * o) / (n - m[:, None] * o)
        p = 1.0 / to
        c = None
        if np.any(to < 0):
            return to, p, m, c, True
        else:
            return to, p, m, c, False

    elif method == 'SHIN':
        r = o.shape[0]
        p = np.zeros_like(o)
        c = np.zeros(r)
        for i in range(r):
            eps = 10**-6
            delta = 10**-6
            eqn = 1
            x = 1.0 / o[i, :]
            while abs(eqn) > eps or np.sum(p[i, :]) > 1:
                p[i, :] = (np.sqrt(c[i]**2 + 4*(1-c[i])*(x**2)/(m[i]+1)) - c[i]) / (2*(1-c[i]))
                eqn = np.sum(p[i, :]) - 1
                pd = (np.sqrt((c[i]+delta)**2 + 4*(1-(c[i]+delta))*(x**2)/(m[i]+1)) - (c[i]+delta)) / (2*(1-(c[i]+delta)))
                eqnd = np.sum(pd, axis=0) - 1
                c[i] -= eqn / ((eqnd - eqn) / delta)
        to = 1.0 / p

    elif method == 'OR':
        r = o.shape[0]
        p = np.zeros_like(o)
        c = np.ones(r)
        for i in range(r):
            eps = 10**-6
            delta = 10**-6
            x = 1.0 / o[i, :]
            eqn = 1
            while abs(eqn) > eps or np.sum(p[i, :]) > 1:
                p[i, :] = x / (c[i] + x - c[i] * x)
                eqn = np.sum(p[i, :]) - 1
                pd = x / ((c[i] + delta) + x - (c[i] + delta) * x)
                eqnd = np.sum(pd, axis=0) - 1
                c[i] -= eqn / ((eqnd - eqn) / delta)
        to = 1.0 / p

    elif method == 'LOG':
        r = o.shape[0]
        p = np.zeros_like(o)
        c = np.ones(r)
        for i in range(r):
            eps = 10**-6
            delta = 10**-6
            x = 1.0 / o[i, :]
            eqn = 1
            while abs(eqn) > eps or np.sum(p[i, :]) > 1:
                p[i, :] = x ** c[i]
                eqn = np.sum(p[i, :]) - 1
                pd = x ** (c[i] + delta)
                eqnd = np.sum(pd, axis=0) - 1
                c[i] -= eqn / ((eqnd - eqn) / delta)
        to = 1.0 / p

    else:
        raise ValueError('Select a method: EM, MPTO, SHIN, OR, LOG')

    return to, p, m, c, False

def update_input_fields():
    for widget in odds_frame.winfo_children():
        widget.destroy()
    odds_entries.clear()
    team_entries.clear()
    odds_5_entries.clear()
    odds_10_entries.clear()
    my_odds_entries.clear()
    advantage_entries.clear()

    try:
        num_outcomes = int(num_outcomes_var.get())
        
        if not odds_frame.grid_slaves(row=0, column=0):
            tk.Label(odds_frame, text="Team/Player").grid(row=0, column=0, padx=10, pady=5)
            tk.Label(odds_frame, text="Odds").grid(row=0, column=1, padx=10, pady=5)
            tk.Label(odds_frame, text="Odds to Beat (LOG + 5%)").grid(row=0, column=2, padx=10, pady=5)
            tk.Label(odds_frame, text="Odds to Beat (LOG + 10%)").grid(row=0, column=3, padx=10, pady=5)
            tk.Label(odds_frame, text="My Odds").grid(row=0, column=4, padx=10, pady=5)
            tk.Label(odds_frame, text="Advantage").grid(row=0, column=5, padx=10, pady=5)

        for i in range(num_outcomes):
            team_entry = tk.Entry(odds_frame, fg="gray", justify="left", width=20)
            default_team_text = f"Selection {i + 1}"
            team_entry.insert(0, default_team_text)
            team_entry.bind("<FocusIn>", lambda event, entry=team_entry, default_text=default_team_text: clear_default_text(entry, default_text))
            team_entry.bind("<FocusOut>", lambda event, entry=team_entry, default_text=default_team_text: restore_default_text(entry, default_text))
            team_entry.grid(row=i + 1, column=0, padx=10, pady=5, sticky="ew")
            team_entries.append(team_entry)

            odds_entry = tk.Entry(odds_frame, fg="gray", justify="left", width=20)
            default_odds_text = f"Odds {i + 1}"
            odds_entry.insert(0, default_odds_text)
            odds_entry.bind("<FocusIn>", lambda event, entry=odds_entry, default_text=default_odds_text: clear_default_text(entry, default_text))
            odds_entry.bind("<FocusOut>", lambda event, entry=odds_entry, default_text=default_odds_text: restore_default_text(entry, default_text))
            odds_entry.grid(row=i + 1, column=1, padx=10, pady=5, sticky="ew")
            odds_entries.append(odds_entry)

            # Create read-only entries for Odds to Beat (5% and 10%)
            odds_5_entry = tk.Entry(odds_frame, state='readonly', width=20)
            odds_5_entry.grid(row=i + 1, column=2, padx=10, pady=5, sticky="ew")
            odds_5_entries.append(odds_5_entry)

            odds_10_entry = tk.Entry(odds_frame, state='readonly', width=20)
            odds_10_entry.grid(row=i + 1, column=3, padx=10, pady=5, sticky="ew")
            odds_10_entries.append(odds_10_entry)

            my_odds_entry = tk.Entry(odds_frame)
            my_odds_entry.grid(row=i + 1, column=4, padx=10, pady=5)
            my_odds_entry.bind("<KeyRelease>", lambda event, idx=i: on_my_odds_change(idx))
            my_odds_entries.append(my_odds_entry)

            advantage_entry = tk.Entry(odds_frame, state='readonly')
            advantage_entry.grid(row=i + 1, column=5, padx=10, pady=5)
            advantage_entries.append(advantage_entry)

        app.update_idletasks()
        app.geometry(f'{app.winfo_width()}x{app.winfo_height()}')

    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid number of outcomes.")

# Function to update My Odds and Advantage in real-time
def on_my_odds_change(index):
    try:
        log_row_index = None
        for row_idx in range(1, result_table.grid_size()[1]):
            method_label = result_table.grid_slaves(row=row_idx, column=0)[0].cget("text")
            if method_label == "LOG":
                log_row_index = row_idx
                break

        if log_row_index is not None:
            log_odds = float(result_table.grid_slaves(row=log_row_index, column=index + 1)[0].cget("text"))
            my_odds = float(my_odds_entries[index].get())
            advantage = (my_odds - log_odds) / log_odds * 100
            advantage_entries[index].config(state='normal')
            advantage_entries[index].delete(0, tk.END)
            advantage_entries[index].insert(0, f"{advantage:.2f}%")
            advantage_entries[index].config(state='readonly')

        if my_odds > log_odds:
            my_odds_entries[index].config(bg='light green', fg="black")
        else:
            my_odds_entries[index].config(bg='light coral', fg="black")
    except ValueError:
        my_odds_entries[index].config(bg='white')
        advantage_entries[index].config(state='normal')
        advantage_entries[index].delete(0, tk.END)
        advantage_entries[index].config(state='readonly')

def bind_my_odds_change():
    for i, entry in enumerate(my_odds_entries):
        entry.bind("<KeyRelease>", lambda event, idx=i: on_my_odds_change(idx))

def update_fields_with_image_data(teams_odds):
    for widget in odds_frame.winfo_children():
        widget.destroy()
    odds_entries.clear()
    team_entries.clear()
    odds_5_entries.clear()
    odds_10_entries.clear()
    my_odds_entries.clear()
    advantage_entries.clear()

    num_outcomes_var.set(str(len(teams_odds)))

    tk.Label(odds_frame, text="Team/Player").grid(row=0, column=0, padx=10, pady=5)
    tk.Label(odds_frame, text="Odds").grid(row=0, column=1, padx=10, pady=5)
    tk.Label(odds_frame, text="Odds to Beat (LOG + 5%)").grid(row=0, column=2, padx=10, pady=5)
    tk.Label(odds_frame, text="Odds to Beat (LOG + 10%)").grid(row=0, column=3, padx=10, pady=5)
    tk.Label(odds_frame, text="My Odds").grid(row=0, column=4, padx=10, pady=5)
    tk.Label(odds_frame, text="Advantage").grid(row=0, column=5, padx=10, pady=5)

    for i, (team, odd) in enumerate(teams_odds):
        team_entry = tk.Entry(odds_frame, width=20)
        team_entry.insert(0, team)
        team_entry.grid(row=i + 1, column=0, padx=10, pady=5, sticky="ew")
        team_entries.append(team_entry)

        odds_entry = tk.Entry(odds_frame, width=20)
        odds_entry.insert(0, str(odd))
        odds_entry.grid(row=i + 1, column=1, padx=10, pady=5, sticky="ew")
        odds_entries.append(odds_entry)

        odds_5_entry = tk.Entry(odds_frame, state='readonly', width=20)
        odds_5_entry.grid(row=i + 1, column=2, padx=10, pady=5, sticky="ew")
        odds_5_entries.append(odds_5_entry)

        odds_10_entry = tk.Entry(odds_frame, state='readonly', width=20)
        odds_10_entry.grid(row=i + 1, column=3, padx=10, pady=5, sticky="ew")
        odds_10_entries.append(odds_10_entry)

        my_odds_entry = tk.Entry(odds_frame)
        my_odds_entry.grid(row=i + 1, column=4, padx=10, pady=5)
        my_odds_entries.append(my_odds_entry)

        advantage_entry = tk.Entry(odds_frame, state='readonly')
        advantage_entry.grid(row=i + 1, column=5, padx=10, pady=5)
        advantage_entries.append(advantage_entry)

    app.update_idletasks()
    app.geometry(f'{app.winfo_width()}x{app.winfo_height()}')

def load_from_spreadsheet():
    file_path = filedialog.askopenfilename(title="Select an Excel file", filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        try:
            df = pd.read_excel(file_path)
            if "Team" in df.columns and "Odds" in df.columns:
                teams_odds = list(zip(df["Team"], df["Odds"]))
                update_fields_with_image_data(teams_odds)
                bind_my_odds_change() 
            else:
                messagebox.showerror("Invalid Format", "The spreadsheet must contain 'Team' and 'Odds' columns.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading spreadsheet: {e}")

def clear_default_text(entry, default_text):
    if entry.get() == default_text:
        entry.delete(0, tk.END)
        entry.config(fg="black")

def restore_default_text(entry, default_text):
    if entry.get() == "":
        entry.insert(0, default_text)
        entry.config(fg="gray")

def highlight_row(event, row_widgets):
    for widget in row_widgets:
        widget.config(bg="#d9eaf7")

def unhighlight_row(event, row_widgets):
    for widget in row_widgets:
        widget.config(bg="white")

# Global variable to track if MPTO is hidden
mpto_hidden = False

# Function to calculate true odds and display results
def calculate_odds():
    global mpto_hidden
    try:
        odds = [float(entry.get()) for entry in odds_entries if "Odds" not in entry.get()]
        odds_array = np.array([odds])

        methods = ['EM', 'MPTO', 'SHIN', 'OR', 'LOG']
        results = {method: [] for method in methods}

        mpto_hidden = False
        log_odds = None

        for method in methods:
            if method == 'MPTO':
                to, p, m, _, has_negative = trueodds(odds_array, method)
                if has_negative:
                    mpto_hidden = True
                    continue
            else:
                to, p, m, _, _ = trueodds(odds_array, method)

            if method == 'LOG':
                log_odds = to[0]

            results[method] = [f"{odd:.4f}" for odd in to[0]]

        for widget in result_table.winfo_children():
            widget.destroy()

        headers = ["Method"]
        for idx, team_entry in enumerate(team_entries):
            if "Selection" in team_entry.get() or not team_entry.get().strip():
                headers.append(f"Selection {idx + 1}")
            else:
                headers.append(team_entry.get())

        # Populate headers in table
        for j, header in enumerate(headers):
            label = tk.Label(result_table, text=header, relief=tk.RIDGE, padx=10, pady=5, bg="#f2f2f2")
            label.grid(row=0, column=j, sticky="nsew")

        # Populate data rows in table
        row_idx = 1
        for method in methods:
            if method == 'MPTO' and mpto_hidden:
                continue

            row_widgets = []
            method_label = tk.Label(result_table, text=method, relief=tk.RIDGE, padx=10, pady=5, bg="white")
            method_label.grid(row=row_idx, column=0, sticky="nsew")
            row_widgets.append(method_label)

            for j, odd in enumerate(results[method]):
                result_label = tk.Label(result_table, text=odd, relief=tk.RIDGE, padx=10, pady=5, bg="white")
                result_label.grid(row=row_idx, column=j + 1, sticky="nsew")
                row_widgets.append(result_label)

            # Bind events to highlight/unhighlight the row
            for widget in row_widgets:
                widget.bind("<Enter>", lambda event, row=row_widgets: highlight_row(event, row))
                widget.bind("<Leave>", lambda event, row=row_widgets: unhighlight_row(event, row))

            row_idx += 1

        if mpto_hidden:
            note_label = tk.Label(result_table, text="* MPTO row is hidden due to negative values.", fg="red", padx=10, pady=5)
            note_label.grid(row=row_idx, column=0, columnspan=len(headers), sticky="w")

        # Calculate and update Odds to Beat (5% and 10%) using LOG odds
        if log_odds is not None:
            for i, log_odd in enumerate(log_odds):
                odds_to_beat_5 = log_odd * 1.05
                odds_to_beat_10 = log_odd * 1.10
                odds_5_entries[i].config(state='normal')
                odds_5_entries[i].delete(0, tk.END)
                odds_5_entries[i].insert(0, f"{odds_to_beat_5:.4f}")
                odds_5_entries[i].config(state='readonly')
                
                odds_10_entries[i].config(state='normal')
                odds_10_entries[i].delete(0, tk.END)
                odds_10_entries[i].insert(0, f"{odds_to_beat_10:.4f}")
                odds_10_entries[i].config(state='readonly')

        export_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

        bind_my_odds_change()

        app.update_idletasks()
        app.geometry("")
        new_height = app.winfo_reqheight()
        new_width = app.winfo_reqwidth()
        app.geometry(f"{new_width}x{new_height}")

    except ValueError:
        pass  # Suppressed error alerts

# Function to export table as an .xlsx file
def export_table_as_excel():
    global mpto_hidden
    headers = ["Method"]
    
    for idx, team_entry in enumerate(team_entries):
        if "Selection" in team_entry.get() or not team_entry.get().strip():
            headers.append(f"Selection {idx + 1}")
        else:
            headers.append(team_entry.get())

    data = []
    
    # Export the rows while skipping the MPTO row if it's hidden
    for row in range(result_table.grid_size()[1] - 1):
        method_label = result_table.grid_slaves(row=row + 1, column=0)[0].cget("text")
        if method_label == 'MPTO' and mpto_hidden:
            continue

        # Skip any "* MPTO row is hidden" message from being added to the export
        if "MPTO row is hidden" in method_label:
            continue

        row_data = [result_table.grid_slaves(row=row + 1, column=col)[0].cget("text") for col in range(len(headers))]
        data.append(row_data)

    # Convert to a pandas DataFrame and save as a spreadsheet
    df = pd.DataFrame(data, columns=headers)
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    
    if file_path:
        df.to_excel(file_path, index=False)
        messagebox.showinfo("Export Success", f"Table exported as an .xlsx file to {file_path}")

# Function to load the image and extract odds data
def load_image_and_extract_odds():
    image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if image_path:
        extracted_text = extract_text_from_image(image_path)
        teams_odds = parse_text(extracted_text)
        update_fields_with_image_data(teams_odds)
        bind_my_odds_change()

app = tk.Tk()
app.title("True Odds Calculator")

num_outcomes_var = tk.StringVar()

odds_entries = []
team_entries = []
odds_5_entries = []
odds_10_entries = []
my_odds_entries = []
advantage_entries = []

tk.Label(app, text="Number of Results:").grid(row=0, column=0, padx=10, pady=5)
num_outcomes_entry = tk.Entry(app, textvariable=num_outcomes_var)
num_outcomes_entry.grid(row=0, column=1, padx=10, pady=5)

generate_button = tk.Button(app, text="Generate Fields", command=update_input_fields)
generate_button.grid(row=0, column=2, padx=10, pady=5)

load_image_button = tk.Button(app, text="Load Image", command=load_image_and_extract_odds)
load_image_button.grid(row=0, column=3, padx=10, pady=5)

import_spreadsheet_button = tk.Button(app, text="Import Spreadsheet", command=load_from_spreadsheet)
import_spreadsheet_button.grid(row=0, column=4, padx=10, pady=5)

calculate_button = tk.Button(app, text="Calculate", command=calculate_odds)
calculate_button.grid(row=0, column=5, padx=10, pady=5)

odds_frame = tk.Frame(app)
odds_frame.grid(row=2, column=0, columnspan=6, padx=10, pady=10)

result_table = tk.Frame(app)
result_table.grid(row=3, column=0, columnspan=6, padx=10, pady=10)

export_button = tk.Button(app, text="Export Table", command=export_table_as_excel)

app.mainloop()