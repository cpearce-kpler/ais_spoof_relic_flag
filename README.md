# ais_spoof_relic_flag
Flags AIS data associated with spoofing relics - perfectly horizontal and vertical lines, plus circles. 

# Dependencies
- Downloaded AIS data sorted by latitude and longitude.

# Run in PowerShell
Here is some example PowerShell code to run the script.
>>     & C:\Users\John Doe> & "C:\anaconda\python.exe" -u "C:\Users\John Doe\Desktop\ais_relic_and_quality_flagger.py" `
>>     --ais-folder "C:\Users\John Doe\Desktop\Data_sets\ais_files\ais_2025" `
>>     --output-dir "C:\Users\John Doe\Desktop\ais_2025_flagged" `
>>     --detector-module-dir "C:\Users\John Doe\Desktop" `
>>     --workers 4
