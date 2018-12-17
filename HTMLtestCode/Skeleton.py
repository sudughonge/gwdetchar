import gwdetchar
from gwdetchar.io import html as htmlio




#file = open('hi.html', 'w')
page = htmlio.new_bootstrap_page(title='scattering')



page.div(class_='container')
page.div(class_='dropdown')
page.button(class_='dropbtn')
page.h1('scattering:')
page.p("This analysis looks for whistles in the data using a technique developed by Patrick Meyers.")
page.div.close()

page.button()


page.button.close()

page.div(class_='panel-heading')
page.p('This is segment' + 'seg')

for x in range(4):
    for y in range(4):
        


# link XML file
page.a()
page.p('XML file link')
page.a.close()


# print state segments
page.p('This analysis was executed over the following segments:')
page.div(class_='panel-group', id_='accordion1')
page.div.close()


with open('hi.html', 'w') as fp:
    fp.write(str(page))


