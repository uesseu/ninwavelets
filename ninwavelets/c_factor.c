int factor(int num, int div){
  while (1){
    int n = num;
    int sq = 3;
    while (n % 2 == 0){
      n /= 2;
    }
    while (sq * sq <= n){
      if (n % sq == 0){ n /= sq; }
      else{ sq += 2; }
    }
    if (n > div){ num += 1; }
    else { return num; }
  }
}
