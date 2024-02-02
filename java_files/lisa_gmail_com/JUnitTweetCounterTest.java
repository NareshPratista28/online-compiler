package lisa_gmail_com;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class JUnitTweetCounterTest {
	private final ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
	  private final PrintStream oPrintStream = System.out;

	  @Before
	  public void setUpStream() {
	    System.setOut(new PrintStream(outputStream));
	  }

	  @After
	  public void restoreStream() {
	    System.setOut(oPrintStream);
	  }

	  @Test
	  public void isTweetCounter() {
	    lisa_gmail_com.TweetCounter.main(null);
	    assertEquals("Tweet Length not same", "83", outputStream.toString());

		String data = lisa_gmail_com.TweetCounter.Tweet(null);
	    assertEquals("tweet in String", "Liz Lemon, ninjas are kind of cool... I just dont know any personally. Get on that.", data);
	  }
}
