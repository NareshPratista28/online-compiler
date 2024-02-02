package lisa_gmail_com; 

public class TweetCounter {
	
	public static String Tweet(String tweet) {
		tweet = "Liz Lemon, ninjas are kind of cool... I just dont know any personally. Get on that.";
		return tweet;
	}
	
	public static void main(String[] args) {
    
	String data = Tweet(null);
    // What's the character count?
    System.out.print(data.length());
  }
}