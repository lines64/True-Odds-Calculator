# True Odds Calculator

True odds calculator written in Python with a nice GUI. Use this calculator to remove the vig/margin from the bookmaker's odds _(use a sharp bookmaker/betting exchange with a lot of liqudity to make sure you have the most accurate odds as possible)_ and see if you can beat their true odds at other bookmakers.

The methods to remove the margin/vig included in the true odds calculator are:

- EM - Equal margin
- MPTO - Margin proportional to odds _(hidden if negative odds are detected)_
- SHIN - Shin
- OR - Odds ratio
- LOG - Logarithmic


Inspired by [WinnerOdds](https://www.winnerodds.com/#lay64).

## Screenshots

![App Screenshot](https://github.com/lines64/True-Odds-Calculator/blob/main/Screenshots/True%20Odds%20Calculator%20Dashboard.png)


### How to run / use

1. **Python Installation**: Ensure you have Python installed on your system. You can download it from [python.org](https://www.python.org/downloads/).
2. **Install Tesseract** _(for image recognition)_. If you are on Windows, you can download it from [here](https://github.com/UB-Mannheim/tesseract/wiki).
3. Install all the dependencies.
   
   ```bash
   pip install pytesseract pillow numpy pandas openpyxl
  
5. Download / clone the repository.
6. Open the True Odds Calculator.pyw file with a text editor and make sure the **Tesseract** path is correct
   
   ```bash
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   
8. Run the **True Odds Calculator.pyw** to open the GUI.
9. There are three ways to insert the betting odds.
    
    - **Manually**: Input the number of results, click generate, input Team/Player _(optional)_ and betting odds, click calculate.
    - **Image**: Load an image, click calculate _(there are two test images in the Test Files folder)_.
    - **Spreadsheet**: Load the spreadsheet, click calculate _(there is an example spreadsheet in the Test Files folder)_.

10. Optional: Insert your own odds into My Odds to see if your bet has any advantage.

    ![App Screenshot](https://github.com/lines64/True-Odds-Calculator/blob/main/Screenshots/True%20Odds%20Calculator%20Advantage.png)

### TODO
    - Create an easy-to-use web version of this calculator

## Feedback

If you have any feedback, please reach out to me at info@lines64.com. Contributions are always welcome!
