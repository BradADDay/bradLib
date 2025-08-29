# bradLib

A simple personal library. Currently contains `csv2tab` and `plotter`.
Will update as I create more utilities.

## `bradLib.csv2tab(file, caption = "", alignment = "")`

|Parameters  | Type |                                                                |
|------------|-----------------------------------------------------------------------|
|`file`      | `str`| the path to the csv file, either relative or absolute          |
|`caption`   | `str`| text to go in the table caption                                |
|`alignment` | `str`| the alignment for each column, must match the number of columns can be: `l`,`r`, or `c` for left, right or centered|
