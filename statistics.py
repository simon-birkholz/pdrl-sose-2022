import matplotlib.pyplot as plt



if __name__ == "__main__":
    fig = plt.figure()
    #ax = fig.add_axes([0, 0, 1, 1])

    swig_times =  []
    python_times = []

    plt.bar(years,tex_counts,label='with .bib', color='tab:green')
    plt.bar(years,remaining_c, bottom=tex_counts, label='Papers', color='tab:blue')
    plt.ylabel('Number of Papers')
    plt.xlabel('Published Year')
    plt.title('Papers per year in crawler arXiv dataset')
    plt.legend()
    #plt.show()
    plt.savefig('arXiv-year-distribution.png')