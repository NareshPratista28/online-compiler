package resh1_gmail_com;

import static org.junit.Assert.*;
import org.junit.Test;

public class JUnitHitungTotalNilaiTest {
    
    @Test
    public void testMainMethodExecution() {
        try {
            HitungTotalNilai.main(new String[]{});
            assertTrue("Main method dapat dijalankan tanpa error", true);
        } catch (Exception e) {
            fail("Main method error: " + e.getMessage());
        }
    }
}