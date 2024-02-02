package test1703_gmail_com; 

import java.util.Arrays;

public class NewsFeed {
	public NewsFeed() {}
	
	public static String[] getTopics() {
		String[] topics = {"Opinion", "Tech", "Science", "Health"};
		return topics;
	}
	
	public static void main(String[] args) {
		NewsFeed samplefeed = new NewsFeed();
		String[] topics = samplefeed.getTopics();

		String topicArray = Arrays.toString(topics);
		System.out.print(topicArray);
	}
}