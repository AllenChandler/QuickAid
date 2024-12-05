@echo off
:: Root directory for data
set "root_dir=J:\projects\QuickAid\data\images"

:: Subdirectories to create
set "folders=Carcinoma Dermatitis Eczema Fungal Lesions Melanocytic Melanoma Psoriasis Tumors Viral" #change for different datasets

:: Create folders in "test" and "valid"
for %%d in (test valid) do (
    for %%f in (%folders%) do (
        mkdir "%root_dir%\%%d\%%f" 2>nul
    )
)

echo Folders created successfully in "test" and "valid".
pause
