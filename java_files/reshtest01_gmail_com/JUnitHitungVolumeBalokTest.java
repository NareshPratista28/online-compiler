package reshtest01_gmail_com;

import org.junit.*;
import java.io.*;

public class JUnitHitungVolumeBalokTest {

    private ByteArrayOutputStream outContent;

    @Before
    public void setUpStreams() {
        outContent = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outContent));
    }

    @After
    public void restoreStreams() {
        System.setOut(System.out);
    }

    @Test
    public void testMainMethodExecution() {
        HitungVolumeBalok.main(new String[]{});
        Assert.assertTrue("Tidak ada kesalahan", outContent.toString().trim().length() > 0);
    }

    @Test
    public void testProgramOutput() {
        HitungVolumeBalok.main(new String[]{});

        String output = outContent.toString();
        Assert.assertEquals("Volume balok positif!", "Volume balok positif!\n" +
                "Panjang: 10.0\n" +
                "Lebar: 5.0\n" +
                "Tinggi: 3.0\n" +
                "Volume balok: 150.0", output.trim());

        // Tampilkan volume
        double expectedVolume = 10 * 5 * 3;
        Assert.assertEquals("Hitung Volume Balok seharusnya menghasilkan nilai volume yang benar!", expectedVolume, Double.parseDouble(output.split("\n")[4].split(": ")[1].trim()), 0);
    }

    @Test
    public void testSpecificFunctionality() {
        HitungVolumeBalok.main(new String[]{});

        double expectedVolume = 10 * 5 * 3;
        Assert.assertEquals("Hitung Volume Balok seharusnya menghasilkan nilai volume yang benar!", expectedVolume, Double.parseDouble(outContent.toString().split("\n")[4].split(": ")[1]), 0);
    }

    @Test
    public void testEdgeCases() {
        HitungVolumeBalok.main(new String[]{});

        Assert.assertTrue("Tidak ada kesalahan", outContent.toString().trim().length() > 0);
    }
}