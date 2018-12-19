import gwdetchar
from gwdetchar.io import html as htmlio




#file = open('hi.html', 'w')
page = htmlio.new_bootstrap_page(title='scattering')

page.div(class_='container')
page.h1('scattering:')
page.p("Title Information")
page.div.close()

page.div(class_='panel-heading')
page.p('This is segment' + 'seg')

for x in range(4):
    page.div(class_='container')
    page.p('Title Information x' + str(x))
    for y in range(4):
        page.div(class_='container')
        page.p('Title Information y' + str(y))
        page.div.close()
        # HOW TO ADD IMAGE:
        file = 'download.png'
        f = open(file)
        page.IMG(src_=file)
        page.p('Just opened image')
    page.div.close()

page.div.close()
        
# HOW TO LINK XML FiLE BELOW:
# link XML file
xml = 'sample.xml'
x = open(xml)
page.a(href_=xml)
page.p('hello')
page.a.close()


# print state segments
page.p('This analysis was executed over the following segments:')
page.div(class_='panel-group', id_='accordion1')
page.div.close()


with open('hi.html', 'w') as fp:
    fp.write(str(page))


