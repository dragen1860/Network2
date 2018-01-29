import sys, select
 

i, o, e = select.select( [sys.stdin], [], [], 1 )

if i:
  print("You said", sys.stdin.readline().strip() )
else:
  print("You said nothing!")