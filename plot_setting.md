## Markdown
- way to open: shift + command + P
- choose open preview to the side

## Basic Setting
- basic line example: http://matplotlib.org/users/pyplot_tutorial.html
- lines2D: https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_marker
- markers: https://matplotlib.org/api/markers_api.html#module-matplotlib.markers

## Boxplot setting
- Input: data and data2
- Description: plot two boxplot in the same category
- Highlight: How to change the box plot color. 
- matplot demo: http://matplotlib.org/examples/pylab_examples/boxplot_demo2.html

## Plot with timeline
- Input: dataframe with two column, one column is date, the other one is some value
- Request: plot the value via the time. 
- Resolution: matplotlib.pyplot.plot_date(dates, values); https://stackoverflow.com/questions/1574088/plotting-time-in-python-with-matplotlib

```python
data = (0, 100, 200)
data2 = (300, 400, 500)

fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
bp = ax1.boxplot([data])
for box in bp['boxes']:
    box.set(color = 'magenta')
for whisker in bp['whiskers']:
    whisker.set(color = 'magenta')
for cap in bp['caps']:
    cap.set(color = 'magenta')

bp2 = ax1.boxplot([data2], 'r')
for box in bp2['boxes']:
    box.set(color = 'cyan')
for whisker in bp2['whiskers']:
    whisker.set(color = 'cyan')
for cap in bp2['caps']:
    cap.set(color = 'cyan')
```