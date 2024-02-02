package lisa_gmail_com; 

public class LuckyFiveSecond {
	
	public static void loop(int number) {
		int i = 1;
	    while (i < number) {
	      System.out.print(i);
	      i++;
	    }
	}
	
	 public static void main(String[] args) {
	      LuckyFiveSecond luck = new LuckyFiveSecond();
	      luck.loop(6);
	 }
}