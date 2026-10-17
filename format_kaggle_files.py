import csv
from datetime import datetime
import os
import sys
import pandas as pd

n = len(sys.argv)

if n == 1:
    print("No input path")
    sys.exit()

inPath = sys.argv[1]
outPath = os.path.join(os.path.dirname(inPath), "formatted_transactions.csv")

currency = dict()
paymentFormat = dict()
bankAcc = dict()
account = dict()

def get_dict_val(name, collection):
    if name in collection:
        val = collection[name]
    else:
        val = len(collection)
        collection[name] = val
    return val

header = "EdgeID,from_id,to_id,Timestamp,Amount Sent,Sent Currency,Amount Received,Received Currency,Payment Format,Is Laundering\n"

firstTs = -1

with open(inPath, 'r', newline='', encoding='utf-8') as infile, open(outPath, 'w', newline='', encoding='utf-8') as writer:
    reader = csv.reader(infile)
    csv_header = next(reader)
    
    # Map header column positions
    col_map = {col.strip(): idx for idx, col in enumerate(csv_header)}
    # Note: 'Account' appears twice: first for 'From Bank' (idx 2), second for 'To Bank' (idx 4)
    # So we use exact positional indices 2 and 4 to avoid name collisions.
    ts_idx = col_map.get("Timestamp", 0)
    from_bank_idx = col_map.get("From Bank", 1)
    from_acc_idx = 2
    to_bank_idx = col_map.get("To Bank", 3)
    to_acc_idx = 4
    amount_rec_idx = col_map.get("Amount Received", 5)
    rec_curr_idx = col_map.get("Receiving Currency", 6)
    amount_paid_idx = col_map.get("Amount Paid", 7)
    pay_curr_idx = col_map.get("Payment Currency", 8)
    pay_fmt_idx = col_map.get("Payment Format", 9)
    isl_idx = col_map.get("Is Laundering", 10)

    writer.write(header)
    for i, row in enumerate(reader):
        datetime_object = datetime.strptime(row[ts_idx], '%Y/%m/%d %H:%M')
        ts = datetime_object.timestamp()
        day = datetime_object.day
        month = datetime_object.month
        year = datetime_object.year
        hour = datetime_object.hour
        minute = datetime_object.minute

        if firstTs == -1:
            startTime = datetime(year, month, day)
            firstTs = startTime.timestamp() - 10

        ts = ts - firstTs

        cur1 = get_dict_val(row[rec_curr_idx], currency)
        cur2 = get_dict_val(row[pay_curr_idx], currency)

        fmt = get_dict_val(row[pay_fmt_idx], paymentFormat)

        fromAccIdStr = row[from_bank_idx] + row[from_acc_idx]
        fromId = get_dict_val(fromAccIdStr, account)

        toAccIdStr = row[to_bank_idx] + row[to_acc_idx]
        toId = get_dict_val(toAccIdStr, account)

        amountReceivedOrig = float(row[amount_rec_idx])
        amountPaidOrig = float(row[amount_paid_idx])

        isl = int(row[isl_idx])

        line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d\n' % \
                    (i, fromId, toId, ts, amountPaidOrig, cur2, amountReceivedOrig, cur1, fmt, isl)

        writer.write(line)

# Sort by Timestamp and write final CSV using pandas (replaces datatable)
formatted = pd.read_csv(outPath)
formatted = formatted.sort_values(by='Timestamp', kind='mergesort')
formatted.to_csv(outPath, index=False)