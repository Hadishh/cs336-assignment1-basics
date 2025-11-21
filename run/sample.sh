awk -v RS='<[|]endoftext[|]>' -v K=10 '
  BEGIN { srand() }
  NF {
    # Current document = record + the separator
    doc = $0 "<|endoftext|>"
    n++

    if (n <= K) {
      sample[n] = doc
    } else {
      # Reservoir sampling: replace with probability K/n
      j = int(1 + rand() * n)
      if (j <= K) sample[j] = doc
    }
  }
  END {
    for (i = 1; i <= K && i <= n; i++) {
      print sample[i]
    }
  }
' "$1" > "$2"
