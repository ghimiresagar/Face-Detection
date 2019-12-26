import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    '''
        Takes the path to the xml file and return panda data
    :param path: path to xml files
    :return: panda data frame with rows and columns
    '''
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):       #gets all the paths to xml
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'annotation/')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('face_labels.csv', index=None)
    print('Successfully converted xml to csv.')


main()