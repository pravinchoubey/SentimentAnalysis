"""
Import sample data for sentiment analysis engine
"""

import predictionio
import argparse
import csv


def import_events(client, file):
  f = open(file, 'r')
  reader = csv.reader(f)
  print("Importing data...")
  rownum = 0
  for row in reader:
    if rownum == 0:
        header = row
        print(header[1],header[2])
    else:
        print(row)
        client.create_event(
           event="train",
           entity_type="phrase",
           entity_id=rownum,
           properties= { "phrase" : row[3], "sentiment": float(row[1]) }
        )
    rownum += 1
  f.close()
  print("%s events are imported." % rownum)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="Import twitter data for sentiment analysis")
  parser.add_argument('--access_key', default='R3p4tqlPvZaO6-K_sZD1dGYYEwnFz3eCovynSM2WeeMECkEBpetTP4ojpobDWt4O')
  parser.add_argument('--url', default="http://sheltered-sierra-30907.herokuapp.com")
  parser.add_argument('--file', default="./data/sample.csv")

  args = parser.parse_args()
  print(args)

  client = predictionio.EventClient(
    access_key=args.access_key,
    url=args.url,
    threads=5,
    qsize=500)
  import_events(client, args.file)
