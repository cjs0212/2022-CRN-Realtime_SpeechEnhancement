package com.example.sample

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.MediaController
import android.widget.VideoView
import androidx.appcompat.app.AppCompatActivity

class SpeechEnhancementActivity : AppCompatActivity() {

    var videoView: VideoView? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_speechenhancement)

        videoView = findViewById(R.id.videoView)

    }

    fun bt1(view: View?) {    // 동영상 선택 누르면 실행됨 동영상 고를 갤러리 오픈
        val intent = Intent()
        intent.type = "video/*"
        intent.action = Intent.ACTION_GET_CONTENT
        startActivityForResult(intent, 101)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) { // 갤러리
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 101) {
            if (resultCode == RESULT_OK) {
                val mc = MediaController(this) // 비디오 컨트롤 가능하게(일시정지, 재시작 등)
                videoView!!.setMediaController(mc)
                val fileUri = data!!.data
                videoView!!.setVideoPath(fileUri.toString()) // 선택한 비디오 경로 비디오뷰에 셋
                videoView!!.start() // 비디오뷰 시작
            }
        }

    }
}