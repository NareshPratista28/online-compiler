package rika_gmail_com;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

public class JUnitMyClassTest {
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
            public void myResultTest() {
                rika_gmail_com.MyClass.main(null);
                assertEquals("MyClass Double Result", "30.5", outputStream.toString());
                
				        double result = rika_gmail_com.MyClass.add(20, 10.5);
                assertEquals("is result = 30.5", "30.5", String.valueOf(result));
            }
}