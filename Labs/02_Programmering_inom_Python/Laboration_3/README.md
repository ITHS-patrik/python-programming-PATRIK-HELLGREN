# Laboration 3 - Linjär klassificering

## Om programmet
Detta program identifierar en linje med formen `y = kx + m` som separerar vår inlästa data så jämnt som möjligt. Datan klassificeras beroende på vilken sida om linjen den befinner sig. Därefter skapas och sparas en ny datafil med både datapunkter och klassificeringar. Avslutningsvis plottas resultatet i en graf.

## Filer
`laboration_3.py` - Huvudprogrammet, beräknar linjer och klassificerar data.<br>
`README.md` - Denna fil, beskriver programmet, dess funktion och användning.<br>
`unlabelled_data.csv` - Innehåller den oklassificerade datan.<br>
`labelled_data.csv` - Innehåller både datapunkter och klassificeringar.<br>
`report.ipynb` - Rapport med kod, grafer samt analys av utfall och klassificeringsmetod.

## Användning
1. Se till att datafilen `unlabelled_data.csv` ligger i samma mapp som programfilen `laboration_3.py`.
2. Exekvera filen `laboration_3.py` för att klassificera datan, generera datafilen `labelled_data.csv` (skrivs över om redan skapad) samt plotta linjen tillsammans med de klassficerade datapunkterna.
3. Öppna filen `report.ipynb` för att ta del av analyser och slutsatser. Exekvera koden i turordning för att visa tillhörande grafer.

## Använda bibliotek
- `Pandas`
- `NumPy`
- `Matplotlib`
- `sys` *(standard)*
- `pathlib` *(standard)*

## Python-version
I detta projekt användes *Python version 3.13.7*.